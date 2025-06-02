#!/usr/bin/env python3
"""
infer.py

A thin Python wrapper around KeySync’s existing shell-based inference pipeline.
It simply invokes `scripts/infer_raw.sh` with the correct arguments, so that we
make zero changes to KeySync’s original code.

Usage example (from another Python file):
    from infer import run_inference
    run_inference(
        video_dir="data/videos",
        audio_dir="data/audios",
        output_dir="my_output_folder",
        keyframe_ckpt="pretrained_models/keyframe_dub.pt",
        interpolation_ckpt="pretrained_models/interpolation_dub.pt",
        compute_until=45,
        fix_occlusion=False,
        position=None,
        start_frame=0
    )
"""

import argparse
import os
import subprocess
import sys

# 1) Locate the KeySync root (assumes this infer.py sits at the same level as keysync/)
_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
_KEYSYNC_ROOT = os.path.join(_THIS_DIR, "keysync")
_SCRIPTS_DIR = os.path.join(_KEYSYNC_ROOT, "scripts")
_INFER_SCRIPT = os.path.join(_SCRIPTS_DIR, "infer_raw.sh")


def run_inference(
    video_dir: str,
    audio_dir: str,
    output_dir: str,
    keyframe_ckpt: str,
    interpolation_ckpt: str,
    compute_until: int = 45,
    fix_occlusion: bool = False,
    position: str = None,
    start_frame: int = 0
) -> None:
    """
    Invokes KeySync’s existing `infer_raw.sh` script with exactly the flags KeySync expects.
    This ensures minimal changes to upstream code.

    Args:
      video_dir: Folder containing input .mp4 files (e.g. "data/videos")
      audio_dir: Folder containing input .wav files (e.g. "data/audios")
      output_dir: Where to write generated frames (will be created if needed)
      keyframe_ckpt: Path to KeySync’s keyframe checkpoint (.pt)
      interpolation_ckpt: Path to KeySync’s interpolation checkpoint (.pt)
      compute_until: Number of seconds of video to process (default: 45)
      fix_occlusion: If True, pass "--fix_occlusion" to the script
      position: Optional "x,y" string for the occlusion mask
      start_frame: Frame index at which to start occlusion (default: 0)

    Raises:
      FileNotFoundError if infer_raw.sh is missing
      RuntimeError if the underlying shell call fails
    """
    # 1) Verify that infer_raw.sh exists
    if not os.path.isfile(_INFER_SCRIPT):
        raise FileNotFoundError(
            f"KeySync’s infer_raw.sh not found at {_INFER_SCRIPT}"
        )

    # 2) Make all provided paths absolute
    video_dir = os.path.abspath(video_dir)
    audio_dir = os.path.abspath(audio_dir)
    output_dir = os.path.abspath(output_dir)
    keyframe_ckpt = os.path.abspath(keyframe_ckpt)
    interpolation_ckpt = os.path.abspath(interpolation_ckpt)

    # 3) Build the command list
    cmd = [
        "/usr/bin/env", "bash", _INFER_SCRIPT,
        "--filelist",     video_dir,
        "--file_list_audio", audio_dir,
        "--output_folder",  output_dir,
        "--keyframes_ckpt", keyframe_ckpt,
        "--interpolation_ckpt", interpolation_ckpt,
        "--compute_until", str(compute_until)
    ]

    # 4) Optionally add occlusion flags, exactly as KeySync expects
    if fix_occlusion:
        cmd.append("--fix_occlusion")
    if position is not None:
        cmd.extend(["--position", position])
    if start_frame != 0:
        cmd.extend(["--start_frame", str(start_frame)])

    # 5) Run the subprocess inside keysync/ directory, keeping current env
    print(f"[KeySync Wrapper] Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=_KEYSYNC_ROOT, env=os.environ.copy())
    if proc.returncode != 0:
        raise RuntimeError(
            f"KeySync inference failed with exit code {proc.returncode}"
        )

    print(f"[KeySync Wrapper] Inference complete. Output at: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Minimal wrapper for KeySync’s infer_raw.sh"
    )
    parser.add_argument(
        "--video_dir", required=True,
        help="Folder containing input .mp4 files (e.g. data/videos)"
    )
    parser.add_argument(
        "--audio_dir", required=True,
        help="Folder containing input .wav files (e.g. data/audios)"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Output folder for generated frames"
    )
    parser.add_argument(
        "--keyframes_ckpt", required=True,
        help="Path to KeySync keyframe checkpoint (.pt)"
    )
    parser.add_argument(
        "--interpolation_ckpt", required=True,
        help="Path to KeySync interpolation checkpoint (.pt)"
    )
    parser.add_argument(
        "--compute_until", type=int, default=45,
        help="Number of seconds of video to process (default: 45)"
    )
    parser.add_argument(
        "--fix_occlusion", action="store_true",
        help="Enable occlusion handling (adds --fix_occlusion)"
    )
    parser.add_argument(
        "--position", type=str, default=None,
        help="Optional x,y for occlusion mask (e.g. '450,450')"
    )
    parser.add_argument(
        "--start_frame", type=int, default=0,
        help="Frame index at which to apply occlusion mask (default: 0)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    run_inference(
        video_dir=args.video_dir,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        keyframe_ckpt=args.keyframes_ckpt,
        interpolation_ckpt=args.interpolation_ckpt,
        compute_until=args.compute_until,
        fix_occlusion=args.fix_occlusion,
        position=args.position,
        start_frame=args.start_frame
    )


if __name__ == "__main__":
    main()
