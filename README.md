# video-compress (CPU-only, minimal)

A reproducible, CPU-only video compression pipeline that:

- Keeps the **same resolution** and **same extension** for outputs  
  - `.mp4 / .mkv / .mov` → **HEVC/H.265 (x265)** with container-aware audio flags  
  - `.avi` → **H.264 (x264)** to keep AVI valid
- Selects settings using **two 6s samples** with **FAST VMAF** (frame subsampling + optional half-res)
- Uses **aligned, full VMAF** for guard & final metrics (timestamp reset, CFR normalize, optional HDR→SDR)
- Protects final quality with a **guard** and a **rescue** re-encode (if needed)
- Can process multiple videos by index using a **ThreadPool**

---

## Prerequisites

- OS: Linux / macOS (Windows via WSL works)
- **FFmpeg** must include:
  - `libx265` (HEVC), `libx264`, and `libvmaf`
- Python 3.9+

> We don’t use any external Python packages — only stdlib + FFmpeg.

---

## Quick Start (all steps run via Python functions — no raw shell)

> All commands below are Python invocations of helper functions in `tasks.py`.

### 1) Verify FFmpeg and encoders/filters

```python
# In your terminal:
# python tasks.py verify_ffmpeg
