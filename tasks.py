#!/usr/bin/env python3
# tasks.py — wrap common shell tasks in Python functions
import os, sys, argparse, shlex, subprocess, configparser, json
from pathlib import Path

# --- helpers ---
def sh(cmd: str, cwd: str = None, check: bool = True, env: dict = None):
    print(f"\n$ {cmd}")
    subprocess.run(cmd, shell=True, check=check, cwd=cwd, env=env)

def load_cfg():
    cfg = configparser.ConfigParser()
    if not Path("config.ini").exists():
        print("config.ini not found.", file=sys.stderr)
        sys.exit(1)
    cfg.read("config.ini")
    return cfg

# --- FFmpeg checks ---
def verify_ffmpeg():
    try:
        out = subprocess.run(["ffmpeg","-hide_banner","-version"], capture_output=True, text=True)
        out.check_returncode()
    except Exception:
        print("FFmpeg not found on PATH.", file=sys.stderr)
        sys.exit(1)
    enc = subprocess.run(["ffmpeg","-hide_banner","-encoders"], capture_output=True, text=True).stdout
    flt = subprocess.run(["ffmpeg","-hide_banner","-filters"],  capture_output=True, text=True).stdout
    has_x265 = "libx265" in enc
    has_x264 = "libx264" in enc
    has_vmaf = "libvmaf" in flt
    print(f"libx265: {has_x265} | libx264: {has_x264} | libvmaf: {has_vmaf}")
    if not has_vmaf or not (has_x265 or has_x264):
        print("Missing required FFmpeg components. Install an FFmpeg build with libx264/libx265/libvmaf.", file=sys.stderr)
        sys.exit(1)

# --- venv helpers ---
def create_venv():
    if Path(".venv").exists():
        print(".venv already exists.")
        return
    sh("python3 -m venv .venv")
    print("\nTo activate the venv:\n  source .venv/bin/activate")

def activate_note():
    print("Activate your venv with:\n  source .venv/bin/activate")

# --- repo introspection ---
def list_indices():
    cfg = load_cfg()
    input_dir = cfg.get("paths","input_dir")
    from compress import list_input_videos  # uses same function in compress.py
    vids = list_input_videos()
    if not vids:
        print(f"No videos in {input_dir}")
        return
    print("Indexed videos:")
    for i, v in enumerate(vids):
        print(f"[{i}] {v}")

# --- run single ---
def run_single(index: int):
    from compress import process_by_index
    res = process_by_index(index)
    print("\n--- SINGLE RESULT ---")
    print(json.dumps(res, indent=2))

# --- run parallel ---
def run_parallel(indices, workers: int = None):
    from compress import process_by_index_parallel
    # convert "0 1 2" → [0,1,2]
    if isinstance(indices, str):
        indices = [int(x) for x in indices.split()]
    res = process_by_index_parallel(indices, max_workers=workers, verbose=True)
    print("\n--- PARALLEL RESULTS ---")
    print(json.dumps(res, indent=2))

# --- git bootstrap helpers ---
def git_init():
    if Path(".git").exists():
        print("Git repo already initialized.")
        return
    sh("git init")
    sh('git add README.md config.ini .gitignore compress.py tasks.py')
    print("Repo initialized and files staged. Next: commit and set remote.")

def git_first_commit(message: str = "Initial commit"):
    sh(f'git commit -m {shlex.quote(message)}')

def git_set_remote(url: str):
    sh(f'git remote add origin {shlex.quote(url)}')

def git_push(branch: str = "main"):
    # ensure branch exists and push upstream
    sh(f'git branch -M {shlex.quote(branch)}')
    sh(f'git push -u origin {shlex.quote(branch)}')

# --- CLI ---
def main():
    ap = argparse.ArgumentParser("tasks.py: helpers to run/install without raw shell")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("verify_ffmpeg")
    sub.add_parser("create_venv")
    sub.add_parser("activate_note")
    sub.add_parser("list_indices")

    p_single = sub.add_parser("run_single")
    p_single.add_argument("--index", type=int, required=True)

    p_par = sub.add_parser("run_parallel")
    p_par.add_argument("--indices", nargs="+", required=True, help="e.g. 0 1 2")
    p_par.add_argument("--workers", type=int, default=None)

    sub.add_parser("git_init")
    p_commit = sub.add_parser("git_first_commit")
    p_commit.add_argument("--message", default="Initial commit")
    p_remote = sub.add_parser("git_set_remote")
    p_remote.add_argument("--url", required=True)
    p_push = sub.add_parser("git_push")
    p_push.add_argument("--branch", default="main")

    args = ap.parse_args()
    if args.cmd == "verify_ffmpeg":
        verify_ffmpeg()
    elif args.cmd == "create_venv":
        create_venv()
    elif args.cmd == "activate_note":
        activate_note()
    elif args.cmd == "list_indices":
        list_indices()
    elif args.cmd == "run_single":
        run_single(args.index)
    elif args.cmd == "run_parallel":
        run_parallel(args.indices, args.workers)
    elif args.cmd == "git_init":
        git_init()
    elif args.cmd == "git_first_commit":
        git_first_commit(args.message)
    elif args.cmd == "git_set_remote":
        git_set_remote(args.url)
    elif args.cmd == "git_push":
        git_push(args.branch)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()