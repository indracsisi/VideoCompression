#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch probe compressor (joblib-parallel), preferring *higher quality at the same CRF*
before dropping to a lower CRF (higher bitrate).

- Mid-LV (100 < Laplacian variance < 230): start with slow/high-quality SVT-AV1
  (preset=4 + TPL + restoration + CDEF + QM) and NO CAR (no downscale).
- Other LV: try fast params at a CRF, and if VMAF is not strong, retry the SAME CRF
  with slow/high-quality params before stepping to a lower CRF.

- Computes fast VMAF (subsampled) and S metric.
- Reports input/output resolution, S, VMAF, C, CRF, laplacian variance.
- No final compressed files are kept (temps cleaned).

Run:
    python video_compress_lv_pref_quality_same_crf.py
"""

import os, sys, math, json, tempfile, shutil, subprocess, pathlib, statistics, time
from typing import List, Tuple, Dict, Optional
from joblib import Parallel, delayed

# ----------------------------
# Config / Paths
# ----------------------------
INPUT_DIR   = "/home/indranil/video_process/good_videos"
EXCEL_PATH  = "/home/indranil/video_process/good_videos_report.xlsx"
CSV_FALLBACK_PATH = "/home/indranil/video_process/good_videos_report.csv"

# Parallel config
N_JOBS = -1  # -1 = use all CPU cores; or set a positive int

# Runtime knobs
FFMPEG_THREADS = "1"  # per-process ffmpeg threads (avoid oversubscription)
VMAF_SUBSAMPLE = 5    # fast VMAF
VMAF_THREADS   = 1

# Target thresholds (we prefer to lift quality at same CRF first)
VMAF_FLOOR_GOOD = 88.0    # “strong” quality we aim for at the same CRF
VMAF_MIN_ALLOW  = 80.0    # minimum allowed (below this S will start collapsing)

# SVT-AV1 parameter sets
# Faster default (quality-efficient but quicker)
SVT_PARAMS_FAST   = "aq-mode=2:enable-qm=1:enable-cdef=1:enable-restoration=1:scd=1"
SVT_PRESET_FAST   = "6"
# Slower, higher quality (use this BEFORE lowering CRF)
SVT_PARAMS_SLOWHQ = "aq-mode=2:enable-qm=1:qm-min=0:qm-max=15:enable-cdef=1:enable-restoration=1:enable-tpl-la=1:scd=1"
SVT_PRESET_SLOW   = "4"

# HEVC (for mov/avi primary) — unchanged paths (we rarely touch these here)
X265_PARAMS_NORMAL = "aq-mode=3:rd=4:limit-sao=1:rect=1:amp=1"
X265_PARAMS_SSIM   = "aq-mode=2:aq-strength=0.8:rd=4:limit-sao=1:rect=1:amp=1:tune=ssim"

# Gentle prefilter
def prefilter_chain():
    return ["hqdn3d=1.0:1.0:4:4", "unsharp=3:3:0.5:3:3:0.0"]

# Primary codec by container
PRIMARY_BY_EXT = {
    ".mp4": "av1",
    ".mkv": "av1",
    ".mov": "hevc",
    ".avi": "hevc",
}

# ----------------------------
# Small utils
# ----------------------------
def run(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr

def ffprobe_json(path: str) -> dict:
    code, out, err = run(["ffprobe","-v","error","-print_format","json","-show_format","-show_streams", path])
    if code != 0:
        raise RuntimeError(f"ffprobe failed: {err}")
    return json.loads(out)

def duration_sec(path: str) -> float:
    return float(ffprobe_json(path)["format"].get("duration", 0.0))

def file_size(path: str) -> int:
    return os.path.getsize(path)

def list_videos(folder: str) -> List[str]:
    vids = []
    for n in sorted(os.listdir(folder)):
        p = os.path.join(folder, n)
        if os.path.isfile(p) and pathlib.Path(p).suffix.lower() in (".mp4",".mkv",".mov",".avi"):
            vids.append(p)
    return vids

def color_flags() -> List[str]:
    return ["-color_primaries","bt709","-color_trc","bt709","-colorspace","bt709","-color_range","tv"]

def probe_encoded_resolution(path: str) -> Tuple[int, int]:
    j = ffprobe_json(path)
    vstreams = [s for s in j.get("streams", []) if s.get("codec_type") == "video"]
    if not vstreams:
        raise RuntimeError("No video stream found when probing encoded file.")
    v = vstreams[0]
    return int(v["width"]), int(v["height"])

# ----------------------------
# Laplacian variance (OpenCV)
# ----------------------------
def compute_laplacian_variance(path: str, sample_frames: int = 20) -> Optional[float]:
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0: cap.release(); return None
    idxs = [max(0, int(i*(total-1)/(sample_frames-1))) for i in range(sample_frames)]
    vals = []
    for frame_idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None: continue
        import cv2 as _cv2
        gray = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY)
        vals.append(_cv2.Laplacian(gray, _cv2.CV_64F).var())
    cap.release()
    return statistics.median(vals) if vals else None

# ----------------------------
# FAST VMAF
# ----------------------------
def build_vmaf_filter(subsample: int, threads: int, log_path: str) -> str:
    opts = [f"n_threads={threads}", "log_fmt=json", f"log_path='{log_path}'"]
    if subsample and subsample > 1:
        opts.append(f"n_subsample={subsample}")
    ref_chain  = "setpts=PTS-STARTPTS,format=yuv420p,setsar=1"
    dist_chain = "setpts=PTS-STARTPTS,format=yuv420p,setsar=1"
    return f"[0:v]{ref_chain}[ref];[1:v]{dist_chain}[dist];[dist][ref]libvmaf=" + ":".join(opts)

def compute_vmaf_fast(ref_path: str, dist_path: str) -> float:
    with tempfile.TemporaryDirectory() as td:
        logp = os.path.join(td, "vmaf.json")
        vfgraph = build_vmaf_filter(VMAF_SUBSAMPLE, VMAF_THREADS, logp)
        cmd = ["ffmpeg","-hide_banner","-loglevel","error","-nostdin",
               "-i", ref_path, "-i", dist_path, "-lavfi", vfgraph, "-f","null","-"]
        code, _, err = run(cmd)
        if code != 0:
            raise RuntimeError(f"fast VMAF failed:\n{err}")
        data = json.load(open(logp))
        try:
            return float(data["pooled_metrics"]["vmaf"]["mean"])
        except:
            vals = [f["metrics"]["vmaf"] for f in data.get("frames",[]) if "metrics" in f and "vmaf" in f["metrics"]]
            if not vals:
                raise RuntimeError("No VMAF frames")
            return sum(vals)/len(vals)

# ----------------------------
# S metric (provided by you)
# ----------------------------
def s_metric(C, V, vmaf_threshold=80, compression_weight=0.7, quality_weight=0.3, soft_threshold_margin=5.0):
    hard_cutoff = vmaf_threshold - soft_threshold_margin
    if V < hard_cutoff:
        return 0.0
    if V < vmaf_threshold:
        soft_pos = (V - hard_cutoff) / soft_threshold_margin
        quality_factor = 0.7 * (soft_pos ** 2)
        if C >= 0.95:
            compression_component = 0.0
        else:
            ratio = 1.0 / C
            compression_component = ((ratio - 1) / 19) ** 1.5 if ratio <= 20 else 1.0 + 0.3 * math.log(ratio / 20.0)
            compression_component = min(1.3, compression_component)
        return min(1.0, compression_component * quality_factor)
    vmaf_excess = V - vmaf_threshold
    quality_component = 0.7 + 0.3 * min(1.0, vmaf_excess / (100.0 - vmaf_threshold))
    if C >= 0.95: compression_component = 0.0
    elif C >= 0.80: compression_component = (1.0 / C - 1.0) ** 2 * 0.4
    else:
        ratio = 1.0 / C
        compression_component = ((ratio - 1.25) / 18.75) ** 1.2 + 0.025 if ratio <= 20 else 1.0 + 0.3 * math.log(ratio / 20.0)
        compression_component = min(1.3, compression_component)
    return min(1.0, compression_weight * compression_component + quality_weight * quality_component)

# ----------------------------
# Strategy selection (LV-aware & “quality-first at same CRF”)
# ----------------------------
def choose_strategy(meta: dict, lap_var: Optional[float]) -> Dict:
    """
    Decide:
      - primary codec (by container)
      - CRF ladder
      - prefilter
      - LV-aware SVT params:
          * If 100<LV<230 and primary=av1 → start with slow/high-quality params (preset=4) and NO CAR
          * Else → start with fast params at each CRF; if VMAF not strong, retry same CRF with slow/high-quality
    """
    ext = meta["ext"]
    primary = PRIMARY_BY_EXT.get(ext, "av1")

    # Unified AV1 ladder (descending quality search)
    crf_ladder = [52, 50, 48, 46, 44, 42] if primary == "av1" else [34, 32, 30, 28, 26]

    # Prefilter (gentle)
    vf_filters = prefilter_chain()

    # LV-aware flags
    in_mid_lv = (lap_var is not None) and (100.0 < lap_var < 230.0)

    return {
        "primary": primary,
        "crf_ladder": crf_ladder,
        "vf_filters": vf_filters,
        "in_mid_lv": in_mid_lv,  # mid band: use slowHQ first & no CAR
        "work_res": None,        # no CAR here; keep native for fair CRF-quality testing
    }

# ----------------------------
# Encoding helpers
# ----------------------------
def encode_trial_av1(src: str, meta: dict, crf: int, work_res: Optional[Tuple[int,int]],
                     preset: str, svt_params: str, vf_filters: Optional[List[str]]) -> Tuple[bool,str,str]:
    ext = meta["ext"]; W,H = meta["W"], meta["H"]
    w_out,h_out = (work_res if work_res else (W,H))
    tmpdir = tempfile.mkdtemp(prefix="enc_")
    out_ext = ".mp4" if ext in (".mp4",".mov") else ".mkv"
    outp   = os.path.join(tmpdir, f"enc{out_ext}")

    vf_chain = []
    if (w_out,h_out)!=(W,H):
        vf_chain.append(f"scale={w_out}:{h_out}:flags=lanczos+accurate_rnd")
    if vf_filters:
        vf_chain.extend(vf_filters)
    vf_chain.append("format=yuv420p10le")
    vf = ",".join(vf_chain)

    cmd = ["ffmpeg","-hide_banner","-loglevel","error","-nostdin","-threads",FFMPEG_THREADS,
           "-y","-i",src,"-vf",vf,*color_flags(),
           "-c:v","libsvtav1","-crf",str(crf),"-preset",preset,"-pix_fmt","yuv420p10le",
           "-svtav1-params",svt_params,
           "-an","-movflags","+faststart", outp]
    code,_,err = run(cmd)
    if code!=0 or not os.path.exists(outp) or file_size(outp)==0:
        shutil.rmtree(tmpdir, True)
        return False, f"encode failed: {err}", ""
    return True, outp, tmpdir

def encode_trial_hevc(src: str, meta: dict, crf: int, work_res: Optional[Tuple[int,int]],
                      x265_params: str, vf_filters: Optional[List[str]]) -> Tuple[bool,str,str]:
    ext = meta["ext"]; W,H = meta["W"], meta["H"]
    w_out,h_out = (work_res if work_res else (W,H))
    tmpdir = tempfile.mkdtemp(prefix="encx_")
    out_ext = ".mp4" if ext in (".mp4",".mov") else ".mkv"
    outp   = os.path.join(tmpdir, f"enc{out_ext}")

    vf_chain = []
    if (w_out,h_out)!=(W,H):
        vf_chain.append(f"scale={w_out}:{h_out}:flags=lanczos+accurate_rnd")
    if vf_filters:
        vf_chain.extend(vf_filters)
    vf_chain.append("format=yuv420p10le")
    vf = ",".join(vf_chain)

    tag = ["-tag:v","hvc1"] if ext in (".mp4",".mov") else []
    cmd = ["ffmpeg","-hide_banner","-loglevel","error","-nostdin","-threads",FFMPEG_THREADS,
           "-y","-i",src,"-vf",vf,*color_flags(), *tag,
           "-c:v","libx265","-crf",str(crf),"-preset","medium","-pix_fmt","yuv420p10le",
           "-x265-params",x265_params,
           "-an", outp]
    code,_,err = run(cmd)
    if code!=0 or not os.path.exists(outp) or file_size(outp)==0:
        shutil.rmtree(tmpdir, True)
        return False, f"encode failed: {err}", ""
    return True, outp, tmpdir

def build_view_like_for_vmaf(encoded_path: str, meta: dict) -> Tuple[bool,str,str]:
    W,H = meta["W"], meta["H"]
    tmpdir = tempfile.mkdtemp(prefix="vlike_")
    outp = os.path.join(tmpdir, "view_like.mp4")
    vf = f"scale={W}:{H}:flags=spline+accurate_rnd,format=yuv420p,setsar=1"
    cmd = ["ffmpeg","-hide_banner","-loglevel","error","-nostdin","-threads",FFMPEG_THREADS,
           "-y","-i",encoded_path,"-vf",vf,*color_flags(),
           "-c:v","libx264","-crf","18","-preset","fast","-pix_fmt","yuv420p","-an", outp]
    code,_,err = run(cmd)
    if code!=0 or not os.path.exists(outp) or file_size(outp)==0:
        shutil.rmtree(tmpdir, True)
        return False, f"reconstruct failed: {err}", ""
    return True, outp, tmpdir

# ----------------------------
# CRF sweep that *prefers* “quality at same CRF”
# ----------------------------
def process_normal(src: str, meta: dict, strat: dict) -> Optional[Dict]:
    size_bytes = file_size(src)
    dur = duration_sec(src)
    crf_ladder = strat["crf_ladder"]
    vf_filters = strat["vf_filters"]
    work_res   = strat["work_res"]
    primary    = strat["primary"]
    in_mid_lv  = strat["in_mid_lv"]

    best, prev_S = None, None

    for crf in crf_ladder:
        tried = []

        if primary == "av1":
            if in_mid_lv:
                # Mid-LV: go SLOW/HQ FIRST at this CRF
                tried.append(("slowhq", SVT_PRESET_SLOW, SVT_PARAMS_SLOWHQ))
            else:
                # Outside mid-LV: try FAST first, then SLOW/HQ at the SAME CRF if needed
                tried.append(("fast",   SVT_PRESET_FAST, SVT_PARAMS_FAST))
                tried.append(("slowhq", SVT_PRESET_SLOW, SVT_PARAMS_SLOWHQ))
        else:
            # HEVC path: just one trial per CRF (unchanged logic)
            tried.append(("hevc", None, None))

        for mode, preset, params in tried:
            # If we did fast first and VMAF already “strong”, we can skip slow to save time.
            # But since we PREFER higher quality at SAME CRF, we’ll only skip slow if VMAF is already high.
            if mode == "slowhq" and best and best.get("CRF")==crf and best.get("VMAF",0)>=VMAF_FLOOR_GOOD:
                continue

            if primary == "av1" and mode in ("fast","slowhq"):
                ok, enc, td = encode_trial_av1(src, meta, crf, work_res, preset, params, vf_filters)
            else:
                ok, enc, td = encode_trial_hevc(src, meta, crf, work_res, X265_PARAMS_NORMAL, vf_filters)

            if not ok:
                continue

            ok2, vlike, td2 = build_view_like_for_vmaf(enc, meta)
            if not ok2:
                shutil.rmtree(td, True); continue

            try:
                V = compute_vmaf_fast(src, vlike)
            except Exception:
                V = 0.0
            C = file_size(enc) / size_bytes
            S = s_metric(C, V)

            try:
                outW, outH = probe_encoded_resolution(enc)
            except Exception:
                outW, outH = (work_res if work_res else (meta["W"], meta["H"]))

            cur = {"file": os.path.basename(src), "duration_sec": dur,
                   "VMAF": V, "C": C, "S": S, "CRF": crf,
                   "If_Downscaled": ("NO" if work_res is None else f"{work_res[0]}x{work_res[1]}"),
                   "note": f"{primary}-{mode}",
                   "in_W": meta["W"], "in_H": meta["H"], "out_W": outW, "out_H": outH}

            # Prefer higher quality at the SAME CRF: if we already have a result at this CRF,
            # take the one with higher VMAF even if C is slightly worse (as long as S doesn’t tank).
            if best is None or S > best["S"] or \
               (best["CRF"] == crf and V > best.get("VMAF",0) and S >= best["S"] - 0.02):
                best = cur

            shutil.rmtree(td2, True); shutil.rmtree(td, True)

            # Early exits:
            # If we are on SLOW/HQ and VMAF already “good” at this CRF, stop; we achieved quality at same CRF.
            if mode == "slowhq" and V >= VMAF_FLOOR_GOOD:
                return best
            # If VMAF well above minimum and S is already strong, stop.
            if S > 0.70 and V >= VMAF_MIN_ALLOW:
                return best

        # If fast at this CRF was poor (VMAF < 80) and slowHQ didn’t rescue (or we’re not in AV1),
        # then and only then step to the next CRF (lower number → higher bitrate).
        # Otherwise, if slowHQ improved but still below 80, we continue to next CRF anyway.
        # (This keeps the “quality-first-at-same-CRF” policy.)
        # No extra code needed—loop just goes to next CRF.

    return best

# ----------------------------
# One video wrapper (joblib worker)
# ----------------------------
def process_one(src: str) -> Dict:
    t0 = time.perf_counter()
    try:
        meta_j = ffprobe_json(src)
        v0 = [s for s in meta_j["streams"] if s.get("codec_type")=="video"][0]
        meta = {"W": int(v0["width"]), "H": int(v0["height"]), "fps": v0.get("avg_frame_rate","24/1"),
                "ext": pathlib.Path(src).suffix.lower()}
        dur = float(meta_j["format"].get("duration", 0.0))

        lap_var = compute_laplacian_variance(src, sample_frames=20)
        strat   = choose_strategy(meta, lap_var)

        print(f"=== {os.path.basename(src)} ===")
        print(f"Laplace (variance): {('NA' if lap_var is None else f'{lap_var:.1f}')}")
        print(f"Strategy: primary={strat['primary']} "
              f"{'slowHQ-first' if strat['in_mid_lv'] and strat['primary']=='av1' else 'fast-then-slowHQ-at-same-CRF'} "
              f"CRFs={strat['crf_ladder']} work_res={'NO' if strat['work_res'] is None else f'{strat['work_res'][0]}x{strat['work_res'][1]}'}")

        best = process_normal(src, meta, strat)

        if best is None:
            best = {"file": os.path.basename(src), "duration_sec": dur,
                    "VMAF": None, "C": None, "S": 0.0, "CRF": None,
                    "If_Downscaled": "NO", "lap_var": lap_var, "note": "no-encode",
                    "in_W": meta["W"], "in_H": meta["H"], "out_W": meta["W"], "out_H": meta["H"]}
        else:
            best["lap_var"] = lap_var
            best.setdefault("in_W", meta["W"])
            best.setdefault("in_H", meta["H"])
            best.setdefault("out_W", best.get("out_W", meta["W"]))
            best.setdefault("out_H", best.get("out_H", meta["H"]))

        dt = time.perf_counter() - t0
        V = best.get("VMAF"); C = best.get("C"); S = best.get("S"); crf = best.get("CRF")
        print(f"[OK {dt:.2f}s] {best.get('file','?')} | "
              f"VMAF={('NA' if V is None else f'{V:.2f}')} | "
              f"C={('NA' if C is None else f'{C:.4f}')} | "
              f"S={('NA' if S is None else f'{S:.3f}')} | CRF={crf} | "
              f"In={best.get('in_W')}x{best.get('in_H')} Out={best.get('out_W')}x{best.get('out_H')}")
        return best
    except Exception as e:
        dt = time.perf_counter() - t0
        print(f"[ERR {dt:.2f}s] {os.path.basename(src)} → {type(e).__name__}: {e}")
        return {"file": os.path.basename(src), "duration_sec": 0.0,
                "VMAF": None, "C": None, "S": 0.0, "CRF": None,
                "If_Downscaled": "NO", "lap_var": None, "note": f"error: {e}",
                "in_W": None, "in_H": None, "out_W": None, "out_H": None}

# ----------------------------
# Excel writer (CSV fallback)
# ----------------------------
def write_table(rows: List[Dict], xlsx_path: str, csv_fallback: str):
    table = []
    for r in rows:
        table.append({
            "Video Name":       r.get("file",""),
            "Duration (s)":     r.get("duration_sec",""),
            "Input W":          r.get("in_W",""),
            "Input H":          r.get("in_H",""),
            "Output W":         r.get("out_W",""),
            "Output H":         r.get("out_H",""),
            "CRF":              r.get("CRF", None),
            "VMAF":             None if r.get("VMAF") is None else float(r["VMAF"]),
            "Compression (C)":  None if r.get("C")   is None else float(r["C"]),
            "S value":          None if r.get("S")   is None else float(r["S"]),
            "If_Downscaled":    r.get("If_Downscaled",""),
            "Note":             r.get("note",""),
            "Laplace Variance": r.get("lap_var",""),
        })
    # Try Excel first
    try:
        import pandas as pd
        df = pd.DataFrame(table, columns=[
            "Video Name","Duration (s)",
            "Input W","Input H","Output W","Output H",
            "CRF","VMAF","Compression (C)","S value",
            "If_Downscaled","Note","Laplace Variance"
        ])
        df.to_excel(xlsx_path, index=False)
        print(f"\nExcel written to: {xlsx_path}")
    except Exception as e:
        # CSV fallback
        try:
            import csv
            with open(csv_fallback, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "Video Name","Duration (s)",
                    "Input W","Input H","Output W","Output H",
                    "CRF","VMAF","Compression (C)","S value",
                    "If_Downscaled","Note","Laplace Variance"
                ])
                writer.writeheader()
                writer.writerows(table)
            print(f"\n(openpyxl/pandas not available) CSV written to: {csv_fallback}")
        except Exception as e2:
            print(f"\n[WARN] Could not write Excel or CSV: {e} / {e2}")

# ----------------------------
# Main (joblib-parallel)
# ----------------------------
def main_parallel_joblib():
    # Avoid thread oversubscription
    os.environ.setdefault("FFMPEG_THREADS", FFMPEG_THREADS)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    paths = list_videos(INPUT_DIR)
    if not paths:
        print(f"[INFO] No videos found in {INPUT_DIR}")
        return

    print(f"[INFO] Found {len(paths)} videos. Parallelizing with joblib, N_JOBS={N_JOBS} ...")
    batch_t0 = time.perf_counter()

    rows = Parallel(n_jobs=N_JOBS, backend="loky", verbose=10)(
        delayed(process_one)(p) for p in paths
    )

    write_table(rows, EXCEL_PATH, CSV_FALLBACK_PATH)
    print(f"[DONE] Total elapsed: {time.perf_counter() - batch_t0:.2f}s")

if __name__ == "__main__":
    main_parallel_joblib()
