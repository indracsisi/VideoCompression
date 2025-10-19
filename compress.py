#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compress.py — Minimal, self-contained pipeline

Features
--------
- Keeps SAME resolution and SAME extension.
- .mp4/.mkv/.mov -> HEVC/H.265 (libx265); .avi -> H.264 (libx264) to keep AVI valid.
- Candidate selection on two 6s samples using FAST VMAF (frame subsampling + optional half-res).
- Aligned FULL VMAF for guard (mid-clip) and final metrics (PTS reset, CFR normalize, optional HDR->SDR).
- Rescue re-encode only if (sample_best_S - final_S) > 0.20 AND final_S <= 0.40.
- Single-video runner (by index) and ThreadPool-based parallel runner.

Usage
-----
# Single
python compress.py --index 0

# Parallel
python compress.py --indices 0 1 2 --workers 2
"""

import os
import sys
import json
import re
import argparse
import configparser
import subprocess
import tempfile
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional, Any

# ============================== Config loader ==============================
CFG = configparser.ConfigParser()
if not os.path.exists("config.ini"):
    print("ERROR: config.ini not found in current directory.", file=sys.stderr)
    sys.exit(1)
CFG.read("config.ini")

INPUT_DIR  = CFG.get("paths", "input_dir")
OUTPUT_DIR = CFG.get("paths", "output_dir")

def _parse_list(s: str, coerce=float):
    items = [t.strip() for t in s.split(",") if t.strip()]
    vals = []
    for it in items:
        if it.lower() == "none":
            vals.append(None)
        else:
            vals.append(coerce(it))
    return vals

CRF_LIST     = _parse_list(CFG.get("search","crf_list"), coerce=int)            # e.g. [26,28,32,34]
VBV_FACTORS  = _parse_list(CFG.get("search","vbv_factors"), coerce=float)       # e.g. [None,0.75,0.5]

SAMPLING_VMAF_SUBSAMPLE      = CFG.getint("sampling","n_subsample", fallback=5) # evaluate every Nth frame
SAMPLING_VMAF_DOWNSCALE_HALF = CFG.getboolean("sampling","half_res", fallback=True)

FFMPEG_THREADS = CFG.get("threads","ffmpeg_threads", fallback="1")               # keep conservative unless tuned
VMAF_THREADS   = CFG.getint("threads","vmaf_threads", fallback=1)

MAX_S_DROP = CFG.getfloat("guard","max_s_drop", fallback=0.20)                   # new rule: allow 0.20 drop

# Fixed knobs (minimal defaults)
ALLOWED_EXTS = {".mp4", ".mkv", ".mov", ".avi"}
X265_PRESET  = "medium"
X264_PRESET  = "medium"
PERCEPTUAL_X265 = {
    "aq-mode": 2, "aq-strength": 1.0, "psy-rd": 1.0, "psy-rdoq": 1.0, "rd": 4,
    "pools": 1, "frame-threads": 1
}
SAMPLE_DURATION     = 6
SAMPLE_OFFSETS_FRAC = [0.2, 0.7]
GENTLE_PREFILTER    = "hqdn3d=1.5:1.5:6:6"  # only used if sample-best S < 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================== Utilities =================================
def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    """
    Run ffmpeg with limited threads. Tweak FFMPEG_THREADS if you have more CPU headroom.
    """
    base = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-threads", FFMPEG_THREADS]
    p = subprocess.Popen(base + cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out, err

def check_ffmpeg() -> None:
    """
    Ensure ffmpeg is present and has libx264/libx265/libvmaf support.
    """
    r = subprocess.run(["ffmpeg","-hide_banner","-version"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg not found on PATH.")
    enc = subprocess.run(["ffmpeg","-hide_banner","-encoders"], capture_output=True, text=True).stdout
    flt = subprocess.run(["ffmpeg","-hide_banner","-filters"],  capture_output=True, text=True).stdout
    has_x265 = "libx265" in enc
    has_x264 = "libx264" in enc
    has_vmaf = "libvmaf" in flt
    print(f"FFmpeg OK | libx265: {has_x265} | libx264: {has_x264} | libvmaf: {has_vmaf}")
    if not has_vmaf or not (has_x265 or has_x264):
        raise RuntimeError("Required components missing: libvmaf and libx265/libx264")

def ffprobe_json(path: str) -> dict:
    p = subprocess.run(["ffprobe","-v","error","-print_format","json","-show_format","-show_streams", path],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {p.stderr}")
    return json.loads(p.stdout)

def get_video_props(path: str) -> dict:
    info    = ffprobe_json(path)
    fmt     = info.get("format", {})
    streams = info.get("streams", [])
    vstreams = [s for s in streams if s.get("codec_type") == "video"]
    if not vstreams:
        raise RuntimeError("No video stream found")
    v = vstreams[0]
    return {
        "duration": float(fmt.get("duration", v.get("duration", 0) or 0)),
        "bit_rate_bps": float(fmt.get("bit_rate", 0)),
        "width": int(v.get("width", 0)),
        "height": int(v.get("height", 0)),
        "pix_fmt": v.get("pix_fmt",""),
        "color_transfer": (v.get("color_transfer") or "").lower(),
        "r_frame_rate": v.get("r_frame_rate") or v.get("avg_frame_rate") or None
    }

def list_input_videos() -> List[str]:
    return list(filter(
        lambda p: pathlib.Path(p).suffix.lower() in ALLOWED_EXTS,
        map(lambda f: str(pathlib.Path(INPUT_DIR) / f), sorted(os.listdir(INPUT_DIR)))
    ))

def out_path_for(src: str, tag: str = "hevc_best") -> str:
    p = pathlib.Path(src)
    return str(pathlib.Path(OUTPUT_DIR) / f"{p.stem}_{tag}{p.suffix.lower()}")

def file_size_bytes(p: str) -> int:
    return os.path.getsize(p)

def S_metric(C: float, VMAF: float) -> float:
    """
    S = 0.8 * (1 - C^1.5) + 0.2 * (VMAF - 80)/20
    """
    return 0.8*(1 - (C**1.5)) + 0.2*((VMAF - 80.0)/20.0)

# ============================== FPS/HDR helpers ============================
def get_source_fps_str(path: str) -> Optional[str]:
    fps = get_video_props(path)["r_frame_rate"]
    return None if (fps in (None, "0/0")) else fps

def is_hdr_like(path: str) -> bool:
    tr = get_video_props(path)["color_transfer"]
    return ("2084" in tr) or ("hlg" in tr) or ("b67" in tr)

# ============================== Candidates =================================
def build_candidates(src_bitrate_bps: float) -> List[Dict]:
    """
    Build (CRF, optional VBV caps) candidates; skip VBV caps if bitrate unknown.
    """
    kbps = src_bitrate_bps/1000.0 if src_bitrate_bps else None
    base = [{"name": f"CRF{crf}", "crf": crf, "maxrate": None, "bufsize": None} for crf in CRF_LIST]
    caps = []
    if kbps:
        for crf in CRF_LIST:
            for f in VBV_FACTORS:
                if f is None: 
                    continue
                caps.append({
                    "name": f"CRF{crf}_cap{int(f*100)}",
                    "crf": crf,
                    "maxrate": int(kbps*f),
                    "bufsize": int(kbps*f)*2
                })
    return base + caps

# ============================== Encoding (extension-aware) ==================
def x265_params_string(d: Dict) -> str:
    return ":".join([f"{k}={v}" for k,v in d.items()])

def encode_extaware(src: str, dst: str, crf: int,
                    maxrate: Optional[int] = None, bufsize: Optional[int] = None,
                    vf: Optional[str] = None) -> None:
    """
    .mp4/.mkv/.mov -> libx265; .avi -> libx264 (to keep AVI valid).
    No scaling; same resolution and same extension preserved.
    """
    ext = pathlib.Path(dst).suffix.lower()
    cmd = ["-i", src]
    if vf:
        cmd += ["-vf", vf]

    if ext == ".avi":
        # x264 for AVI
        v_args = ["-c:v","libx264","-preset", X264_PRESET, "-crf", str(crf), "-pix_fmt","yuv420p"]
        a_args = ["-c:a","libmp3lame","-b:a","192k"]
        if maxrate and bufsize:
            v_args += ["-maxrate", f"{maxrate}k", "-bufsize", f"{bufsize}k"]
        cmd += v_args + a_args + [dst]
    else:
        # x265 for mp4/mkv/mov
        v_args = ["-c:v","libx265","-preset", X265_PRESET, "-crf", str(crf),
                  "-x265-params", x265_params_string(PERCEPTUAL_X265),
                  "-pix_fmt","yuv420p","-tag:v","hvc1"]
        if ext == ".mp4":
            a_args = ["-c:a","aac","-b:a","128k","-movflags","+faststart"]
        elif ext == ".mkv":
            a_args = ["-c:a","copy"]
        elif ext == ".mov":
            a_args = ["-c:a","aac","-b:a","192k","-movflags","+faststart"]
        else:
            a_args = ["-c:a","copy"]
        if maxrate and bufsize:
            v_args += ["-maxrate", f"{maxrate}k", "-bufsize", f"{bufsize}k"]
        cmd += v_args + a_args + [dst]

    code, out, err = run_cmd(cmd)
    if code != 0:
        raise RuntimeError(f"Encode failed ({ext}): {err}")

# ============================== Sample clips ===============================
def make_clip(src: str, dst: str, start: float, dur: float) -> None:
    """
    Build a temporary reference clip (video-only, yuv420p) for fair sampling comparisons.
    """
    code, out, err = run_cmd([
        "-ss", str(start), "-t", str(dur), "-i", src,
        "-map","0:v:0",
        "-c:v","libx264","-preset","ultrafast","-crf","18",
        "-pix_fmt","yuv420p",
        "-an",
        dst
    ])
    if code != 0:
        raise RuntimeError(f"Clip creation failed for {src}: {err}")

def sample_clips(src: str, duration: int, offsets_frac: List[float]) -> Tuple[tempfile.TemporaryDirectory, List[str]]:
    props = get_video_props(src)
    offs  = [max(0.0, props["duration"]*f - duration/2) for f in offsets_frac]
    tmpd  = tempfile.TemporaryDirectory()
    paths = [os.path.join(tmpd.name, f"ref_{i}.mp4") for i in range(len(offs))]
    for i in range(len(offs)):
        make_clip(src, paths[i], offs[i], duration)
    return tmpd, paths

# ============================== VMAF (aligned) =============================
def _aligned_legs(src_for_norm: str, use_half_res: bool) -> Tuple[str, str]:
    """
    Build per-leg filter chains: PTS reset, optional tonemap, CFR normalize, optional half-res, yuv420p, SAR=1.
    """
    fps = get_source_fps_str(src_for_norm)
    hdr = is_hdr_like(src_for_norm)
    chain = ["setpts=PTS-STARTPTS"]
    if hdr:
        chain += ["zscale=t=linear","tonemap=hable","zscale=matrix=bt709:transfer=bt709:primaries=bt709"]
    if fps:
        chain += [f"fps=fps={fps}"]
    if use_half_res:
        chain += ["scale=iw/2:ih/2:flags=bicubic"]
    chain += ["format=yuv420p","setsar=1"]
    return ",".join(chain)+"[ref]", ",".join(chain)+"[dist]"

def vmaf_mean_aligned_fast(ref: str,
                           dist: str,
                           src_for_norm: Optional[str] = None,
                           n_subsample: int = SAMPLING_VMAF_SUBSAMPLE,
                           half_res: bool = SAMPLING_VMAF_DOWNSCALE_HALF,
                           vmaf_threads: int = VMAF_THREADS) -> float:
    """
    Fast VMAF for sampling: uses subsampling and optional half-res to rank candidates quickly.
    """
    with tempfile.TemporaryDirectory() as td:
        logp = os.path.join(td, "vmaf.json")
        srcn = src_for_norm or ref
        ref_leg, dist_leg = _aligned_legs(srcn, use_half_res=half_res)
        opts = [f"n_threads={vmaf_threads}", "log_fmt=json", f"log_path='{logp}'"]
        if n_subsample and n_subsample > 1:
            opts.append(f"n_subsample={n_subsample}")
        fg = f"[0:v]{ref_leg};[1:v]{dist_leg};[dist][ref]libvmaf=" + ":".join(opts)
        code, out, err = run_cmd(["-i", ref, "-i", dist, "-map","0:v:0","-map","1:v:0", "-lavfi", fg, "-f","null","-"])
        if code != 0:
            raise RuntimeError(f"VMAF (fast) failed: {err}")
        with open(logp,"r") as f:
            data = json.load(f)
        try:
            return float(data["pooled_metrics"]["vmaf"]["mean"])
        except Exception:
            frames = data.get("frames", [])
            vals = [fr["metrics"]["vmaf"] for fr in frames if "metrics" in fr and "vmaf" in fr["metrics"]]
            if not vals: 
                raise RuntimeError("VMAF JSON missing values")
            return sum(vals)/len(vals)

def vmaf_mean_aligned_full(ref: str,
                           dist: str,
                           src_for_norm: Optional[str] = None,
                           vmaf_threads: int = VMAF_THREADS) -> float:
    """
    Full, aligned VMAF for guard and final metrics (no subsampling; no half-res).
    """
    with tempfile.TemporaryDirectory() as td:
        logp = os.path.join(td, "vmaf.json")
        srcn = src_for_norm or ref
        ref_leg, dist_leg = _aligned_legs(srcn, use_half_res=False)
        fg = f"[0:v]{ref_leg};[1:v]{dist_leg};[dist][ref]libvmaf=n_threads={vmaf_threads}:log_fmt=json:log_path='{logp}'"
        code, out, err = run_cmd(["-i", ref, "-i", dist, "-map","0:v:0","-map","1:v:0", "-lavfi", fg, "-f","null","-"])
        if code != 0:
            raise RuntimeError(f"VMAF (full) failed: {err}")
        with open(logp,"r") as f:
            data = json.load(f)
        try:
            return float(data["pooled_metrics"]["vmaf"]["mean"])
        except Exception:
            frames = data.get("frames", [])
            vals = [fr["metrics"]["vmaf"] for fr in frames if "metrics" in fr and "vmaf" in fr["metrics"]]
            if not vals: 
                raise RuntimeError("VMAF JSON missing values")
            return sum(vals)/len(vals)

# ============================== Sampling evaluation =======================
def evaluate_on_samples(src: str, candidates: List[Dict]) -> List[Dict]:
    """
    Evaluate candidates on two 6s reference clips using FAST VMAF.
    Returns list sorted by (avg_S, avg_vmaf_fast) desc.
    """
    tmp_ref, ref_clips = sample_clips(src, SAMPLE_DURATION, SAMPLE_OFFSETS_FRAC)
    orig_sizes = [file_size_bytes(p) for p in ref_clips]
    avg_orig   = sum(orig_sizes)/len(orig_sizes)

    pairs = [(candidates[i // len(ref_clips)], ref_clips[i % len(ref_clips)])
             for i in range(len(candidates) * len(ref_clips))]

    raw = []
    for cand, ref_clip in pairs:
        with tempfile.TemporaryDirectory() as td:
            dist = os.path.join(td, f"dist_{cand['name']}.mp4")   # temp container ok
            encode_extaware(ref_clip, dist, crf=cand["crf"],
                            maxrate=cand["maxrate"], bufsize=cand["bufsize"], vf=None)
            q = vmaf_mean_aligned_fast(ref_clip, dist, src_for_norm=src)
            s = file_size_bytes(dist)
        raw.append( (cand["name"], cand["crf"], cand["maxrate"], cand["bufsize"], q, s) )

    # aggregate
    agg: Dict[str, Dict[str, List[float]]] = {}
    for name, crf, maxrate, bufsize, q, sz in raw:
        if name not in agg:
            agg[name] = {"q": [], "sz": [], "crf": crf, "maxrate": maxrate, "bufsize": bufsize}
        agg[name]["q"].append(q); agg[name]["sz"].append(sz)

    results = []
    for name, dat in agg.items():
        avg_q  = sum(dat["q"])/len(dat["q"])
        avg_sz = sum(dat["sz"])/len(dat["sz"])
        C      = min(1.0, max(0.0, avg_sz/avg_orig))
        S      = S_metric(C, avg_q)
        results.append({"name": name, "crf": dat["crf"], "maxrate": dat["maxrate"], "bufsize": dat["bufsize"],
                        "avg_vmaf_fast": avg_q, "avg_C": C, "avg_S": S})
    results.sort(key=lambda r: (r["avg_S"], r["avg_vmaf_fast"]), reverse=True)
    return results

# ============================== Guard (mid-clip, FULL VMAF) ================
def verify_choice_on_midclip(src: str, chosen: Dict) -> Tuple[float, float, float, Dict]:
    """
    Encode a 6s mid-clip with chosen settings; compute FULL aligned VMAF/C/S.
    If VMAF < 80, first remove VBV caps; else lower CRF by 2 (min 18). Up to 2 attempts.
    """
    props = get_video_props(src)
    start = max(0.0, props["duration"]*0.5 - SAMPLE_DURATION/2)
    with tempfile.TemporaryDirectory() as td:
        ref  = os.path.join(td, "ref_mid.mp4")
        dist = os.path.join(td, "dist_mid.mp4")
        make_clip(src, ref, start, SAMPLE_DURATION)

        adj = dict(chosen)
        for _ in range(2):
            encode_extaware(ref, dist, crf=adj["crf"], maxrate=adj["maxrate"], bufsize=adj["bufsize"], vf=None)
            V = vmaf_mean_aligned_full(ref, dist, src_for_norm=src)
            C = min(1.0, max(0.0, file_size_bytes(dist) / file_size_bytes(ref)))
            S = S_metric(C, V)
            if V >= 80.0:
                return V, C, S, adj
            # First: drop VBV caps if present; then: lower CRF by 2
            if adj.get("maxrate") or adj.get("bufsize"):
                adj["maxrate"], adj["bufsize"] = None, None
            else:
                adj["crf"] = max(18, adj["crf"] - 2)

        V = vmaf_mean_aligned_full(ref, dist, src_for_norm=src)
        C = min(1.0, max(0.0, file_size_bytes(dist) / file_size_bytes(ref)))
        S = S_metric(C, V)
        return V, C, S, adj

# ============================== Final encode + report ======================
def full_encode_and_report(src: str, best_result: Dict) -> Dict:
    """
    Full encode using guard-adjusted settings; compute final FULL aligned VMAF/C/S.
    Rescue only if (sample_best_S - final_S) > MAX_S_DROP AND final_S <= 0.40.
    """
    # Guard pass
    Vg, Cg, Sg, adj = verify_choice_on_midclip(src, best_result)
    if adj["crf"] != best_result["crf"] or adj.get("maxrate") != best_result.get("maxrate"):
        print(f"Guard adjusted → CRF {best_result['crf']} → {adj['crf']}, caps={bool(adj.get('maxrate'))}")

    dst = out_path_for(src, "hevc_best")
    use_pref = (best_result["avg_S"] < 0.5)
    vf = GENTLE_PREFILTER if use_pref else None

    # Full encode
    encode_extaware(src=src, dst=dst, crf=adj["crf"], maxrate=adj["maxrate"], bufsize=adj["bufsize"], vf=vf)

    # Final metrics
    V = vmaf_mean_aligned_full(src, dst, src_for_norm=src)
    C = min(1.0, max(0.0, file_size_bytes(dst) / file_size_bytes(src)))
    S = S_metric(C, V)

    # New rule: rescue only if drop > 0.20 AND final S <= 0.40
    drop = best_result["avg_S"] - S
    need_rescue = (drop > MAX_S_DROP) and (S <= 0.40)

    if need_rescue:
        print(f"Rescue: Final S ({S:.3f}) dropped {drop:.3f} (> {MAX_S_DROP}) AND S<=0.40. Retrying with lower CRF & no caps.")
        rescue = dict(adj)
        rescue["crf"] = max(18, adj["crf"] - 2)
        rescue["maxrate"], rescue["bufsize"] = None, None

        dst2 = out_path_for(src, "hevc_rescue")
        encode_extaware(src=src, dst=dst2, crf=rescue["crf"], maxrate=None, bufsize=None, vf=vf)

        V2 = vmaf_mean_aligned_full(src, dst2, src_for_norm=src)
        C2 = min(1.0, max(0.0, file_size_bytes(dst2) / file_size_bytes(src)))
        S2 = S_metric(C2, V2)

        if S2 > S:
            print(f"Rescue improved S: {S:.3f} → {S2:.3f}. Using rescue output.")
            try:
                os.replace(dst2, dst)
            except Exception:
                os.remove(dst); os.rename(dst2, dst)
            V, C, S = V2, C2, S2
        else:
            try: os.remove(dst2)
            except: pass
    else:
        if drop > MAX_S_DROP and S > 0.40:
            print(f"No rescue: drop {drop:.3f} > {MAX_S_DROP} but S={S:.3f} > 0.40 → keep as-is.")

    print("\n=== FINAL RESULTS ===")
    print("Output:", dst)
    print(f"Final VMAF: {V:.2f}")
    print(f"Final C   : {C:.4f}")
    print(f"Final S   : {S:.4f}")
    print("Targets:")
    print(" - VMAF > 80 =>", "OK" if V > 80 else "NOT MET")
    print(" - S > 0.5   =>", "OK" if S > 0.5 else "NOT MET")

    return {"output": dst, "VMAF": V, "C": C, "S": S, "prefilter_used": use_pref}

# ============================== Drivers ===================================
def process_by_index(idx: int) -> Dict:
    """
    Lists inputs, selects by index, evaluates candidates on samples, 
    runs a full encode with the best, and prints final metrics.
    """
    check_ffmpeg()

    videos = list_input_videos()
    if not videos:
        raise RuntimeError(f"No videos in {INPUT_DIR} with {sorted(ALLOWED_EXTS)}")

    print("Indexed videos:")
    for i, v in enumerate(videos):
        print(f"[{i}] {v}")

    if idx < 0 or idx >= len(videos):
        raise IndexError(f"Index {idx} out of range (0..{len(videos)-1})")

    src = videos[idx]
    print("\n==============================")
    print(f"[{idx}] Processing: {src}")

    props = get_video_props(src)
    candidates = build_candidates(props["bit_rate_bps"])

    sample_results = evaluate_on_samples(src, candidates)
    best = sample_results[0]
    print("\nBest on samples (fast VMAF):")
    print(best)

    final = full_encode_and_report(src, best)

    print("\n=== BEST S (final) ===")
    print(f"S = {final['S']:.4f}, VMAF = {final['VMAF']:.2f}, C = {final['C']:.4f}")
    return {"samples_best": best, "final": final}

def process_by_index_parallel(indices: List[int], max_workers: Optional[int] = None, verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Run process_by_index() for a list of indices in parallel using a thread pool.
    """
    if not isinstance(indices, list) or not all(isinstance(i, int) for i in indices):
        raise ValueError("`indices` must be a list of integers.")
    if max_workers is None:
        max_workers = min(len(indices), max(1, (os.cpu_count() or 4)//2))

    results: List[Dict[str, Any]] = [None] * len(indices)

    def _worker(idx: int, pos: int):
        if verbose:
            print(f"[THREAD] start idx={idx}")
        try:
            out = process_by_index(idx)
            if verbose:
                print(f"[THREAD] done  idx={idx} (OK)")
            return (pos, out)
        except Exception as e:
            if verbose:
                print(f"[THREAD] done  idx={idx} (ERROR: {e})")
            return (pos, {"error": str(e)})

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(_worker, idx, pos): (pos, idx) for pos, idx in enumerate(indices)}
        for fut in as_completed(future_map):
            pos, _idx = future_map[fut]
            try:
                pos_ret, payload = fut.result()
                results[pos_ret] = payload
            except Exception as e:
                results[pos] = {"error": f"Unhandled exception for idx={_idx}: {e}"}

    for i, r in enumerate(results):
        if r is None:
            results[i] = {"error": f"No result produced for idx={indices[i]}"}

    if verbose:
        ok = sum(1 for r in results if isinstance(r, dict) and "error" not in r)
        err = len(results) - ok
        print(f"\n[SUMMARY] Completed {len(results)} tasks: {ok} OK, {err} error(s).")
    return results

# ============================== CLI =======================================
def _main():
    ap = argparse.ArgumentParser("video-compress (minimal)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--index", type=int, help="Single index to process")
    g.add_argument("--indices", nargs="+", type=int, help="Multiple indices to process in parallel")
    ap.add_argument("--workers", type=int, default=None, help="ThreadPool workers (parallel mode)")
    args = ap.parse_args()

    if args.index is not None:
        res = process_by_index(args.index)
        print(res)
    else:
        res = process_by_index_parallel(args.indices, max_workers=args.workers)
        print(res)

if __name__ == "__main__":
    _main()