import os, sys, math, json, tempfile, shutil, subprocess, pathlib, statistics, time
from typing import List, Tuple, Dict, Optional

# ----------------------------
# Config / Paths
# ----------------------------
INPUT_DIR   = "/home/indranil/video_process/test_videos"
EXCEL_PATH  = "/home/indranil/video_process/test_videos_report.xlsx"
CSV_FALLBACK_PATH = "/home/indranil/video_process/test_videos_report.csv"

# ----------------------------
# Optional dependencies
# ----------------------------
try:
    import cv2
except Exception:
    cv2 = None

try:
    import pandas as pd
except Exception:
    pd = None

# ----------------------------
# LV bands (aligned to your observation)
# ----------------------------
LV_LOW  = 100     # flat-ish below this
LV_HIGH = 230     # 100..229 = mid-detail; >=230 = high-detail

# ----------------------------
# Search space (band-specific)
# ----------------------------
# Working scales (internal downscale) — final is restored to original size in the SAME encode.
SCALES_FLAT      = [1.00, 0.90, 0.83]
CRF_COARSE_FLAT  = [52, 48, 44, 40, 36]
REFINE_FLAT      = [-2, -1, +1, +2]

SCALES_MID       = [1.00, 0.95, 0.90]
CRF_COARSE_MID   = [50, 48, 46, 44, 42]
REFINE_MID       = [-2, -1, +1]

SCALES_DETAILED  = [1.00, 0.95]
CRF_COARSE_HIGH  = [52, 50, 48, 44, 40, 38, 36]
REFINE_HIGH      = [-2, -1, +1, +2]

def bounded_crf(v: int, lo=30, hi=60):
    return max(lo, min(hi, v))

# ----------------------------
# Encoder / metrics settings
# ----------------------------
VMAF_MODEL = "vmaf_v0.6.1.json"
VMAF_SUBSAMPLE = 5

# Color / format flags
COLOR_FLAGS = [
    "-color_primaries", "bt709",
    "-color_trc", "bt709",
    "-colorspace", "bt709",
    "-color_range", "tv"
]

# SVT-AV1 params (safe, version-tolerant)
# We'll toggle enable-qm by band; keep scene-cut detect on.
def build_svt_params(flat_safe: bool) -> str:
    base = ["aq-mode=2", "scd=1"]
    base.append("enable-qm=0" if flat_safe else "enable-qm=1")
    return ":".join(base)

# ----------------------------
# S metric (exactly as provided)
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
# Utils
# ----------------------------
def run(cmd: List[str], check=True, capture=False) -> Optional[str]:
    if capture:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check, text=True)
        return r.stdout
    else:
        subprocess.run(cmd, check=check)

def ffprobe_meta(path: str) -> Dict[str, float]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,duration",
        "-of", "json", path
    ]
    out = run(cmd, capture=True)
    data = json.loads(out)
    s = data["streams"][0]
    w = int(s.get("width", 0))
    h = int(s.get("height", 0))
    dur = float(s.get("duration", 0.0))
    fr = s.get("avg_frame_rate", "0/0")
    try:
        num, den = fr.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 0.0
    except Exception:
        fps = 0.0
    ext = pathlib.Path(path).suffix.lower()
    return {"w": w, "h": h, "fps": fps, "dur": dur, "ext": ext}

def compute_lv(path: str, samples=20) -> Optional[float]:
    if cv2 is None:
        return None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if frame_count == 0:
        cap.release()
        return None
    idxs = [int(i * frame_count / (samples + 1)) for i in range(1, samples + 1)]
    vals = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lv = cv2.Laplacian(gray, cv2.CV_64F).var()
        vals.append(lv)
    cap.release()
    return statistics.median(vals) if vals else None

def file_size_bytes(path: str) -> Optional[int]:
    try:
        return os.path.getsize(path)
    except Exception:
        return None

# ----------------------------
# VMAF (robust: REF first, scale2ref)
# ----------------------------
def run_vmaf(ref_path: str, dist_path: str) -> float:
    import tempfile, json, subprocess, os
    def _run(cmd):
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return p.returncode, p.stdout, p.stderr
    with tempfile.TemporaryDirectory() as td:
        logp = os.path.join(td, "vmaf.json")
        vf = (
            "[0:v]setpts=PTS-STARTPTS,format=yuv420p,setsar=1[ref];"
            "[1:v]setpts=PTS-STARTPTS,format=yuv420p,setsar=1[dist_pre];"
            "[dist_pre][ref]scale2ref=flags=bicubic[dist][ref_sized];"
            f"[dist][ref_sized]libvmaf=log_fmt=json:log_path='{logp}':n_threads=1:n_subsample={VMAF_SUBSAMPLE}"
        )
        cmd = [
            "ffmpeg","-hide_banner","-loglevel","error","-nostdin",
            "-i", ref_path, "-i", dist_path,
            "-lavfi", vf,
            "-f","null","-"
        ]
        code, out, err = _run(cmd)
        if code != 0:
            raise RuntimeError(f"VMAF failed (code {code}).\n{err}")
        with open(logp, "r") as f:
            data = json.load(f)
        try:
            return float(data["pooled_metrics"]["vmaf"]["mean"])
        except Exception:
            vals = [fr["metrics"]["vmaf"] for fr in data.get("frames",[]) if "metrics" in fr and "vmaf" in fr["metrics"]]
            if not vals:
                raise RuntimeError("No VMAF frames in log.")
            return sum(vals)/len(vals)

# ----------------------------
# Filter chain (single encode: down -> optional denoise -> up -> optional unsharp)
# ----------------------------
def build_vf_chain(orig_w: int, orig_h: int, scale_factor: float, lv: Optional[float]) -> Tuple[str, bool]:
    sf = scale_factor
    flat_band = (lv is not None and lv < LV_LOW)
    mid_band  = (lv is not None and LV_LOW <= lv < LV_HIGH)

    down = "null"
    up   = f"scale={orig_w}:{orig_h}:flags=spline+accurate_rnd+full_chroma_int"
    denoise = "null"
    sharp   = "null"

    if sf < 0.999:
        work_w = max(2, int(round(orig_w * sf)))
        work_h = max(2, int(round(orig_h * sf)))
        down = f"scale={work_w}:{work_h}:flags=spline+accurate_rnd+full_chroma_int"

    if flat_band:
        denoise = "hqdn3d=0.8:0.8:4.0:4.0"
    elif mid_band:
        denoise = "hqdn3d=0.6:0.6:3.0:3.0"

    if mid_band:
        sharp = "unsharp=5:5:0.35:3:3:0.20"

    vf = ",".join([down, denoise, up, sharp, "setsar=1"])
    flat_safe = (flat_band or mid_band)  # QM off if LV<230
    return vf, flat_safe

# ----------------------------
# SVT-AV1 encode (band-aware speed: preset 3 for mid band)
# ----------------------------
def svt_av1_encode_single(src: str, out_path: str, crf: int, vf_chain: str, flat_safe: bool, slow_mode: bool):
    svt_params = build_svt_params(flat_safe)
    preset = "3" if slow_mode else "5"   # slower for better RD in mid band
    gop = "300"                          # long GOP; SCD still inserts keyframes at cuts

    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-vf", vf_chain,
        "-map", "0:v:0",
        "-c:v", "libsvtav1",
        "-pix_fmt", "yuv420p10le",
        "-crf", str(crf),
        "-preset", preset,
        "-g", gop,
        "-movflags", "+faststart",
        "-svtav1-params", svt_params,
        *COLOR_FLAGS,
        "-an",
        out_path
    ]
    run(cmd)

# ----------------------------
# One trial (scale, CRF) → single encode → compute S
# ----------------------------
def trial_one(tmpdir: str, src: str, meta: Dict[str, float], crf: int, sf: float, lv: Optional[float]) -> Dict[str, Optional[float]]:
    W, H = meta["w"], meta["h"]
    vf, flat_safe = build_vf_chain(W, H, sf, lv)

    # Slow mode for the LV 100..229 band (problem zone), else normal
    slow_mode = (lv is None) or (LV_LOW <= lv < LV_HIGH)

    out_path = os.path.join(tmpdir, f"enc_s{int(sf*100)}_crf{crf}.mp4")
    svt_av1_encode_single(src, out_path, crf, vf, flat_safe=flat_safe, slow_mode=slow_mode)

    # Build view-likes @ original size and compute VMAF (REF first!)
    ref_view  = os.path.join(tmpdir, "ref_view.mp4")
    dist_view = os.path.join(tmpdir, f"dist_s{int(sf*100)}_crf{crf}.mp4")
    build_viewlike(src, W, H, ref_view)
    build_viewlike(out_path, W, H, dist_view)

    vmaf = run_vmaf(ref_view, dist_view)
    orig_sz = file_size_bytes(src) or 1
    enc_sz  = file_size_bytes(out_path) or None
    C = (enc_sz / orig_sz) if enc_sz else None
    S = s_metric(C, vmaf) if (C is not None and vmaf is not None) else None

    return {
        "final": out_path,
        "CRF": crf, "Scale": sf, "VMAF": vmaf, "C": C, "S": S
    }

# ----------------------------
# Band-aware search (grid + local refine) → pick max S
# ----------------------------
def search_best(tmpdir: str, src: str, meta: Dict[str, float], lv: Optional[float]) -> Dict[str, Optional[float]]:
    if lv is None:
        band = "mid"
        scales, crf_coarse, refine = SCALES_MID, CRF_COARSE_MID, REFINE_MID
    elif lv < LV_LOW:
        band = "flat"
        scales, crf_coarse, refine = SCALES_FLAT, CRF_COARSE_FLAT, REFINE_FLAT
    elif lv < LV_HIGH:
        band = "mid"
        scales, crf_coarse, refine = SCALES_MID, CRF_COARSE_MID, REFINE_MID
    else:
        band = "high"
        scales, crf_coarse, refine = SCALES_DETAILED, CRF_COARSE_HIGH, REFINE_HIGH

    trials: List[Dict[str, Optional[float]]] = []

    for sf in scales:
        coarse_res = []
        for crf in crf_coarse:
            t = trial_one(tmpdir, src, meta, crf, sf, lv)
            trials.append(t); coarse_res.append(t)

        # Local refinement around the best coarse CRF at this scale
        best_c = max(coarse_res, key=lambda x: (x["S"] or -1.0))
        base = best_c["CRF"]
        neighbors = sorted(set(bounded_crf(base + d) for d in refine))
        for crf in neighbors:
            t = trial_one(tmpdir, src, meta, crf, sf, lv)
            trials.append(t)

    # Pick maximum S; tie-break to smaller C, then larger VMAF
    trials.sort(key=lambda x: ((x["S"] or -1.0), -(x["C"] or 1.0), (x["VMAF"] or -1.0)), reverse=True)
    best = trials[0] if trials else {"final": None, "CRF": None, "Scale": None, "VMAF": None, "C": None, "S": None}
    best["band"] = band
    return best

# ----------------------------
# Per-video processing
# ----------------------------
def process_one(path: str) -> Dict[str, Optional[float]]:
    meta = ffprobe_meta(path)
    if pathlib.Path(path).suffix.lower() != ".mp4":
        return {"file": os.path.basename(path), "path": os.path.abspath(path), "note": "Skipped: not .mp4"}

    lv = compute_lv(path)
    with tempfile.TemporaryDirectory() as tmp:
        best = search_best(tmp, path, meta, lv)

        # Move best output near the input
        if best["final"]:
            out_name = f"{pathlib.Path(path).stem}_av1s{int(best['Scale']*100)}_crf{best['CRF']}.mp4"
            out_path = os.path.join(os.path.dirname(path), out_name)
            try:
                shutil.move(best["final"], out_path)
            except Exception:
                shutil.copy2(best["final"], out_path)
        else:
            out_path = None

    return {
        "file": os.path.basename(path),
        "path": os.path.abspath(path),
        "duration": meta["dur"],
        "LV": lv,
        "band": best.get("band"),
        "CRF": best.get("CRF"),
        "Scale": best.get("Scale"),
        "VMAF": best.get("VMAF"),
        "C": best.get("C"),
        "S": best.get("S"),
        "output": out_path,
        "If_Downscaled": "YES" if (best.get("Scale") and best["Scale"] < 0.999) else "NO",
        "primary": "av1",
        "note": ""
    }

# ----------------------------
# Results writer
# ----------------------------
def write_table(rows: List[Dict[str, Optional[float]]], xlsx_path: str, csv_path: str):
    if not rows:
        return
    if pd is None:
        import csv
        keys = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[INFO] Wrote CSV: %s" % csv_path)
        return
    df = pd.DataFrame(rows)
    try:
        df.to_excel(xlsx_path, index=False)
        print(f"[INFO] Wrote Excel: %s" % xlsx_path)
    except Exception:
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Wrote CSV (fallback): %s" % csv_path)

# ----------------------------
# Main (stream paths; .mp4 only; one-by-one)
# ----------------------------
def main():
    if not os.path.isdir(INPUT_DIR):
        print(f"[ERROR] INPUT_DIR not found: {INPUT_DIR}")
        return

    start = time.time()
    all_results: List[Dict[str, Optional[float]]] = []
    video_paths: List[str] = []

    entries = sorted(os.scandir(INPUT_DIR), key=lambda e: e.name)
    found_any = False
    for e in entries:
        if not e.is_file():
            continue
        if pathlib.Path(e.name).suffix.lower() != ".mp4":
            continue

        found_any = True
        vpath = e.path
        video_paths.append(vpath)

        try:
            res = process_one(vpath)
            all_results.append(res)
            vmaf_str = "NA" if (res.get("VMAF") is None) else f"{res['VMAF']:.2f}"
            c_str    = "NA" if (res.get("C")    is None) else f"{res['C']:.4f}"
            s_str    = "NA" if (res.get("S")    is None) else f"{res['S']:.3f}"
            crf_str  = str(res.get("CRF", "NA"))
            sc_str   = "NA" if (res.get("Scale") is None) else f"{res['Scale']:.2f}"
            print(f"{res['file']} | LV={res.get('LV')} | band={res.get('band')} | "
                  f"VMAF={vmaf_str} | C={c_str} | S={s_str} | CRF={crf_str} | "
                  f"Scale={sc_str} | If_Downscaled={res.get('If_Downscaled','NO')} | output={res.get('output')}")
        except Exception as ex:
            print(f"[ERROR] {vpath}: {ex}")

    if not found_any:
        print(f"No .mp4 videos in {INPUT_DIR}")
        return

    print("\nJSON Results:")
    print(json.dumps(all_results, indent=2))
    write_table(all_results, EXCEL_PATH, CSV_FALLBACK_PATH)
    print("Elapsed Time = ", round(time.time() - start, 2), "s")

# ----------------------------
if __name__ == "__main__":
    main()
