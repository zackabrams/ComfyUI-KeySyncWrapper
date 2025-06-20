#!/usr/bin/env python3
"""
infer.py

Thin wrapper around KeySync’s own `scripts/infer_raw.sh`.
We do NOT modify any KeySync code—just forward arguments exactly as upstream expects.
"""

import os
import subprocess
import argparse

# 1) Locate the KeySync root (assumes infer.py sits alongside a subfolder “keysync/”)
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
    start_frame: int = 0,
    decoding_t: int = None,
    chunk_size: int = None
) -> None:
    """
    Calls KeySync’s infer_raw.sh script with exactly the flags it expects.

    Args:
      video_dir: Directory containing input .mp4 (e.g., run_dir with input.mp4)
      audio_dir: Directory containing input .wav (e.g., run_dir with input.wav)
      output_dir: Directory where KeySync will place output PNG frames
      keyframe_ckpt: Path to keyframe_dub.pt
      interpolation_ckpt: Path to interpolation_dub.pt
      compute_until: Number of seconds to process
      fix_occlusion: If True, pass `--fix_occlusion`
      position: Optional x,y for occlusion
      start_frame: Frame index to start occlusion
      decoding_t: Number of frames decoded at a time (VRAM usage)
      chunk_size: Size of chunks for processing (memory management)
    """
    if not os.path.isfile(_INFER_SCRIPT):
        raise FileNotFoundError(f"KeySync’s infer_raw.sh not found at {_INFER_SCRIPT}")

    # Absolute paths
    video_dir = os.path.abspath(video_dir)
    audio_dir = os.path.abspath(audio_dir)
    output_dir = os.path.abspath(output_dir)
    keyframe_ckpt = os.path.abspath(keyframe_ckpt)
    interpolation_ckpt = os.path.abspath(interpolation_ckpt)
    
    # KeySync expects either a file path or a directory.
    # If we're passing directories, we should look for the actual files
    video_path = video_dir
    audio_path = audio_dir
    
    # Check if video_dir is a directory containing input.mp4 (look in subdirectories too)
    if os.path.isdir(video_dir):
        # First check direct path
        video_file = os.path.join(video_dir, "input.mp4")
        if os.path.exists(video_file):
            video_path = video_file
        else:
            # Check in videos subdirectory
            video_file = os.path.join(video_dir, "videos", "input.mp4")
            if os.path.exists(video_file):
                video_path = video_file
            else:
                # Fall back to directory (KeySync will scan for .mp4 files)
                print(f"[KeySync Wrapper] Warning: No input.mp4 found in {video_dir} or {video_dir}/videos, passing directory")
    
    # Check if audio_dir is a directory containing input.wav (look in subdirectories too) 
    if os.path.isdir(audio_dir):
        # First check direct path
        audio_file = os.path.join(audio_dir, "input.wav")
        if os.path.exists(audio_file):
            audio_path = audio_file
        else:
            # Check in audios subdirectory
            audio_file = os.path.join(audio_dir, "audios", "input.wav")
            if os.path.exists(audio_file):
                audio_path = audio_file
            else:
                # Fall back to directory (KeySync will scan for .wav files)
                print(f"[KeySync Wrapper] Warning: No input.wav found in {audio_dir} or {audio_dir}/audios, passing directory")

    cmd = [
        "/usr/bin/env", "bash", _INFER_SCRIPT,
        "--file_list",          video_path,
        "--file_list_audio",    audio_path,
        "--output_folder",      output_dir,
        "--keyframes_ckpt",     keyframe_ckpt,
        "--interpolation_ckpt", interpolation_ckpt,
        "--compute_until",      str(compute_until)
    ]
    
    # Add optimization parameters if provided
    if decoding_t is not None:
        cmd.extend(["--decoding_t", str(decoding_t)])
    if chunk_size is not None:
        cmd.extend(["--chunk_size", str(chunk_size)])

    if fix_occlusion:
        cmd.append("--fix_occlusion")
    if position is not None:
        cmd.extend(["--position", position])
    if start_frame != 0:
        cmd.extend(["--start_frame", str(start_frame)])

    print(f"[KeySync Wrapper] Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=_KEYSYNC_ROOT, env=os.environ.copy())
    if proc.returncode != 0:
        raise RuntimeError(f"KeySync inference failed with exit code {proc.returncode}")
    print(f"[KeySync Wrapper] Inference complete. Outputs at: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Thin wrapper for KeySync’s infer_raw.sh")
    parser.add_argument("--video_dir", required=True, help="Directory containing input .mp4 files")
    parser.add_argument("--audio_dir", required=True, help="Directory containing input .wav files")
    parser.add_argument("--output_dir", required=True, help="Directory for output PNG frames")
    parser.add_argument("--keyframes_ckpt", required=True, help="Path to keyframe_dub.pt")
    parser.add_argument("--interpolation_ckpt", required=True, help="Path to interpolation_dub.pt")
    parser.add_argument("--compute_until", type=int, default=45, help="Seconds of video to process")
    parser.add_argument("--fix_occlusion", action="store_true", help="Enable occlusion handling")
    parser.add_argument("--position", type=str, default=None, help="Optional x,y for occlusion mask")
    parser.add_argument("--start_frame", type=int, default=0, help="Frame index where occlusion starts")
    parser.add_argument("--decoding_t", type=int, default=None, help="Number of frames decoded at a time")
    parser.add_argument("--chunk_size", type=int, default=None, help="Size of chunks for processing")
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
        start_frame=args.start_frame,
        decoding_t=args.decoding_t,
        chunk_size=args.chunk_size
    )


if __name__ == "__main__":
    main()
