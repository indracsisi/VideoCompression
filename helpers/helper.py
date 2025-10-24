# helpers/helper.py
import os, json, tempfile, pathlib, subprocess, shutil
from typing import List, Tuple, Dict

# ---------- Shell & probe ----------

def run(cmd: List[str]) -> tuple[int, str, str]:
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

# ---------- Paths & names ----------

def ext_of(path: str) -> str:
    return pathlib.Path(path).suffix.lower()

def stem_of(path: str) -> str:
    return pathlib.Path(path).stem

def out_like(src: str, out_dir: str | None, tag: str) -> str:
    p = pathlib.Path(src)
    out_dir = p.parent if out_dir is None else pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{p.stem}_{tag}{p.suffix.lower()}")

# ---------- VMAF (fast, normalized) ----------

def _norm_bt709_chain() -> list[str]:
    # Normalize to SDR bt709 and set SAR=1. VMAF branch itself forces yuv420p.
    return ["zscale=matrix=bt709:transfer=bt709:primaries=bt709", "setsar=1"]

def build_vmaf_filter(n_subsample: int, threads: int) -> str:
    ref  = ["setpts=PTS-STARTPTS", *_norm_bt709_chain(), "format=yuv420p", "setsar=1", "[ref]"]
    dist = ["setpts=PTS-STARTPTS", *_norm_bt709_chain(), "format=yuv420p", "setsar=1", "[dist]"]
    opts = [f"n_threads={threads}", "log_fmt=json", "log_path='${LOGP}'"]
    if n_subsample and n_subsample > 1:
        opts.append(f"n_subsample={n_subsample}")
    return f"[0:v]{','.join(ref)};[1:v]{','.join(dist)};[dist][ref]libvmaf=" + ":".join(opts)

def compute_vmaf_fast(src: str, dist: str, *, subsample: int = 5, threads: int = 1) -> float:
    with tempfile.TemporaryDirectory() as td:
        logp = os.path.join(td, "vmaf.json")
        filt = build_vmaf_filter(subsample, threads).replace("${LOGP}", logp)
        code, _, err = run([
            "ffmpeg","-hide_banner","-loglevel","error","-nostdin",
            "-i", src, "-i", dist, "-lavfi", filt, "-f", "null", "-"
        ])
        if code != 0:
            raise RuntimeError(f"VMAF failed: {err}")
        data = json.load(open(logp))
        try:
            return float(data["pooled_metrics"]["vmaf"]["mean"])
        except Exception:
            vals = [f["metrics"]["vmaf"] for f in data.get("frames", []) if "metrics" in f and "vmaf" in f["metrics"]]
            if not vals: raise RuntimeError("VMAF JSON empty")
            return sum(vals) / len(vals)

# ---------- Metric S ----------

def s_metric(C: float, vmaf: float, w1: float = 0.7, w2: float = 0.3) -> float:
    return w1 * (1 - (C ** 1.5)) + w2 * ((vmaf - 80.0) / 20.0)

# ---------- Sampling & clips ----------

def pick_samples(total: float, sample_len: float) -> list[tuple[float, float]]:
    if total < 30.0:
        return []
    n = 2 if total < 90.0 else (3 if total < 180.0 else 4)
    span = max(0.0, total - sample_len)
    starts = [i * (span / (n - 1)) if n > 1 else span / 2 for i in range(n)]
    return [(max(0.0, min(s, total - sample_len)), sample_len) for s in starts]

def extract_clip(src: str, start: float, dur: float, out_ext: str = ".mkv") -> str:
    td = tempfile.mkdtemp()
    dst = os.path.join(td, f"clip_{int(start)}_{int(start+dur)}{out_ext}")
    fmt = "matroska" if out_ext == ".mkv" else "mp4"
    code, _, err = run([
        "ffmpeg","-hide_banner","-loglevel","error","-nostdin",
        "-ss", str(start), "-t", str(dur), "-i", src,
        "-map", "0", "-c", "copy",
        "-fflags","+genpts","-copyts","-avoid_negative_ts","make_zero",
        "-f", fmt, dst
    ])
    if code != 0 or not os.path.exists(dst) or file_size(dst) == 0:
        shutil.rmtree(td, ignore_errors=True)
        raise RuntimeError(f"extract_clip failed: {err}")
    return dst

# ---------- Prefilter (gentle) ----------

def gentle_prefilter(enabled: bool) -> list[str]:
    return (["atadenoise=0.5:0.5:0.5:0.5", "gradfun=thr=0.5"] if enabled else [])
