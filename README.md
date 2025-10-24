# VideoCompression

Content-adaptive, single-video and batch compression that **keeps the same resolution and the same extension**, with **fast VMAF** scoring and an **S metric** (size/quality balance). Codec is chosen by container:

- **.mp4 / .mkv → AV1 (SVT-AV1, 10-bit)**
- **.mov → HEVC (x265 main10, tagged `hvc1`)**
- **.avi → H.264 (x264, 8-bit)**

The pipeline performs **per-title CRF selection** using short samples, applies a **very light prefilter** (optional) to squeeze bitrate without visible loss, then does a **final full encode** with quality-oriented settings. Scoring uses **fast VMAF** and a composite **S** metric.

---

## Repository layout

VideoCompression/
├─ compress.py # single-video CLI
├─ batch_compress.py # batch runner (scans Data/testbench_small)
├─ config.ini # encoder presets, CRF ladders, VMAF settings, S weights
├─ helpers/
│ ├─ init.py
│ └─ helper.py # ffprobe/run, VMAF, sampling, S metric, utils
├─ Data/
│ ├─ testbench_small/ # put input videos here
│ │ └─ .gitkeep
│ └─ testbench_small_output/ # outputs + final videos
│ └─ .gitkeep
├─ requirements.txt
├─ README.md
└─ .gitignore


---

## How it works

### 1) Choose codec by extension
- `.mp4` / `.mkv` → `libsvtav1` (10-bit `yuv420p10le`)
- `.mov` → `libx265` main10 + `-tag:v hvc1` (Apple compatibility)
- `.avi` → `libx264` high profile (8-bit, AVI-friendly)

### 2) Sampling (per-title CRF selection)
- If **duration < 30 s** → skip sampling; probe CRFs on the **full clip** with faster settings.
- Else take **2–4 × 6 s samples** across the timeline (2 for <90 s, 3 for <180 s, 4 otherwise).
- For each CRF in a **ladder**, encode the samples, compute **fast VMAF** (subsampled), compression ratio **C**, and:

S = 0.7 * (1 - C^1.5) + 0.3 * ((VMAF - 80) / 20)

- Pick the **CRF with best S**, preferring **VMAF ≥ 80** and **C ≤ 0.60** when attainable.

### 3) Final encode
- Encode the **whole video** at the chosen CRF with **slower/quality** presets.
- Compute **fast VMAF**, **C**, and **S** on the final output.

### Gentle prefilter (optional)
- `atadenoise` (very light) + `gradfun` debanding help the encoder, typically allowing +1–2 CRF at similar perceived quality.
- Toggle in `config.ini` (`filter.prefilter_on`).

---

## Requirements

### FFmpeg build with the right codecs/filters
Your `ffmpeg` must be compiled with:
- `--enable-libvmaf` (VMAF)
- `--enable-libsvtav1` (AV1)
- `--enable-libx265` (HEVC)
- `--enable-libx264` (H.264)

Check:
```bash
ffmpeg -hide_banner -filters  | grep libvmaf
ffmpeg -hide_banner -encoders | egrep "svtav1|libx265|libx264"

Key knobs:

VMAF speed: vmaf.subsample=5 (fast). Set 1 for full VMAF.

Prefilter: filter.prefilter_on=true/false.

Quality vs speed: presets (preset_sample/preset_final).

CRF ladders: tweak per codec family.

How to run
A) Single video - 
# output goes next to the input (filename_final.ext)
python3 compress.py /path/to/video.mp4

# or choose output directory
python3 compress.py /path/to/video.mkv --outdir ./Data/testbench_small_output

Put your input videos into:

Data/testbench_small/


Supported extensions: .mp4, .mkv, .mov, .avi

Run: python3 batch_compress.py


Outputs:

Compressed videos in Data/testbench_small_output/

Logs:
Data/testbench1_small_log_01.txt
Data/testbench1_small_log_01.xlsx (columns: File Name, Size (MB), Final VMAF, Final Compression (C), Final S, Output Path)

Tips & notes

AV1 is used for MP4/MKV and is slower than H.264/HEVC but compresses better at the same quality.
MOV uses HEVC main10 and is tagged hvc1 for Apple player compatibility.
AVI uses H.264 (HEVC/AV1 in AVI is not reliable).
If sources are HDR or non-bt709, the VMAF branches are normalized to bt709 for fair scoring. Encodes remain SDR (no resolution change).
For speed, keep vmaf.subsample=5, use the provided presets, and keep the prefilter on unless it hurts your content.

Troubleshooting

“ffmpeg not built with libvmaf” → Install an FFmpeg build with --enable-libvmaf.

AV1 encoder missing (libsvtav1) → Install an FFmpeg with SVT-AV1 enabled.

Final VMAF < 80 but file is small → include lower CRFs earlier in your ladder, or disable the prefilter for very sharp/grainy content.

Too slow → increase presets (preset_final=6), keep vmaf.subsample=5, or disable prefilter.

RUN Commands
============
# (optional) venv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Single file
python3 compress.py ./Data/testbench_small/sample.mp4 --outdir ./Data/testbench_small_output

# Batch
python3 batch_compress.py