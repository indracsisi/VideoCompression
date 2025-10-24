#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch compressor:
- Reads all videos from Data/testbench_small/
- Writes outputs to Data/testbench_small_output/
- Logs results to Data/testbench1_small_log_01.txt and .xlsx

Depends on compress.py (imports its functions) and helpers/.
"""

import os
import sys
import time
import pathlib
import traceback
import pandas as pd

from compress import (
    load_config, ext_of, duration_sec, file_size,
    choose_crf, final_encode, final_metrics
)

ALLOWED = {".mp4", ".mkv", ".mov", ".avi"}

# Paths inside repo
REPO_ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR  = REPO_ROOT / "Data"
IN_DIR    = DATA_DIR / "testbench_small"
OUT_DIR   = DATA_DIR / "testbench_small_output"

LOG_TXT   = DATA_DIR / "testbench1_small_log_01.txt"
LOG_XLSX  = DATA_DIR / "testbench1_small_log_01.xlsx"


def ensure_dirs():
    IN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def list_videos():
    files = []
    for p in sorted(IN_DIR.iterdir()):
        if p.is_file() and ext_of(str(p)) in ALLOWED:
            files.append(p)
    return files


def main():
    ensure_dirs()
    cfg = load_config()

    rows = []
    started = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(LOG_TXT, "w", encoding="utf-8") as ftxt:
        ftxt.write(f"Batch started: {started}\n")
        ftxt.write(f"Input dir : {IN_DIR}\n")
        ftxt.write(f"Output dir: {OUT_DIR}\n\n")

        vids = list_videos()
        if not vids:
            ftxt.write("No input videos found in Data/testbench_small.\n")
            print("No input videos found in Data/testbench_small.")
            return

        for idx, vp in enumerate(vids, 1):
            src = str(vp)
            try:
                ftxt.write(f"[{idx}/{len(vids)}] {src}\n")
                print(f"[{idx}/{len(vids)}] {src}")

                ext = ext_of(src)
                # choose CRF via samples / per-title selection
                choice = choose_crf(cfg, src, ext)
                crf = choice["crf"]

                # final encode to OUT_DIR
                out_path = str((OUT_DIR / f"{vp.stem}_final{vp.suffix}").resolve())
                # final_encode in compress.py creates name from src; we want OUT_DIR target:
                # So we temporarily symlink/copy? Simpler: let final_encode write next to src, then move.
                tmp_out = None
                try:
                    tmp_out = final_encode(cfg, src, None, crf)  # writes next to src
                    # move to OUT_DIR/name_final.ext
                    os.replace(tmp_out, out_path)
                except Exception:
                    # if move fails, still attempt metrics on tmp_out
                    if tmp_out and os.path.exists(tmp_out):
                        out_path = tmp_out
                    else:
                        raise

                # final metrics
                m = final_metrics(cfg, src, out_path)

                # log
                size_mb = file_size(out_path) / (1024*1024)
                line = (f"  CRF={crf}  VMAF={m['VMAF']:.2f}  C={m['C']:.4f}  S={m['S']:.3f}\n"
                        f"  -> {out_path}  ({size_mb:.2f} MB)\n")
                ftxt.write(line + "\n")
                print(line)

                rows.append({
                    "File Name": vp.name,
                    "Size (MB)": round(size_mb, 2),
                    "Final VMAF": round(m["VMAF"], 2),
                    "Final Compression (C)": round(m["C"], 4),
                    "Final S": round(m["S"], 3),
                    "Output Path": out_path
                })

            except Exception as e:
                err = f"  ERROR: {e}\n{traceback.format_exc()}\n"
                ftxt.write(err + "\n")
                print(err)

        # write Excel
        if rows:
            df = pd.DataFrame(rows,
                              columns=["File Name","Size (MB)","Final VMAF","Final Compression (C)","Final S","Output Path"])
            df.to_excel(LOG_XLSX, index=False)
            ftxt.write(f"\nWrote Excel log: {LOG_XLSX}\n")
            print(f"Wrote Excel log: {LOG_XLSX}")

        ended = time.strftime("%Y-%m-%d %H:%M:%S")
        ftxt.write(f"\nBatch ended: {ended}\n")
        print(f"Batch ended: {ended}")


if __name__ == "__main__":
    main()
