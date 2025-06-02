import cv2
import torch
import numpy as np
import soundfile as sf
import subprocess
import os
from typing import List

def prepare_video_for_keysync(images: torch.Tensor, output_path: str, fps: float = 25.0):
    """
    Convert ComfyUI image tensor to video file for KeySync processing
    """
    # Convert tensor to numpy and handle format
    if isinstance(images, torch.Tensor):
        frames = images.cpu().numpy()
    else:
        frames = np.array(images)
    
    # Handle different tensor formats
    if frames.ndim == 4:
        # Check if channel-first [N, C, H, W] or channel-last [N, H, W, C]
        if frames.shape[1] in [1, 3, 4]:  # Channel-first
            frames = np.transpose(frames, (0, 2, 3, 1))
    
    # Ensure uint8 format
    if frames.dtype == np.float32 or frames.dtype == np.float64:
        if frames.max() <= 1.0:
            frames = (frames * 255).astype(np.uint8)
        else:
            frames = frames.astype(np.uint8)
    
    # Get video dimensions
    height, width = frames.shape[1:3]
    num_frames = frames.shape[0]
    
    # Use FFmpeg to create video (more reliable than OpenCV for various codecs)
    ffmpeg_cmd = [
        'ffmpeg', '-y',  # -y to overwrite output file
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'rgb24',
        '-r', str(fps),
        '-i', '-',  # Read from stdin
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',  # High quality
        output_path
    ]
    
    try:
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Write frames to FFmpeg
        for frame in frames:
            process.stdin.write(frame.tobytes())
        
        process.stdin.close()
        process.wait()
        
        if process.returncode != 0:
            stderr = process.stderr.read().decode()
            raise RuntimeError(f"FFmpeg error: {stderr}")
            
    except Exception as e:
        raise RuntimeError(f"Video encoding failed: {e}")

def prepare_audio_for_keysync(audio: dict, output_path: str):
    """
    Convert ComfyUI audio to WAV file for KeySync processing
    """
    if not isinstance(audio, dict):
        raise ValueError("Expected audio dict from VideoHelperSuite")
    
    waveform = audio.get('waveform')
    sample_rate = audio.get('sample_rate', 44100)
    
    if waveform is None:
        raise ValueError("No waveform found in audio dict")
    
    # Convert to numpy if tensor
    if isinstance(waveform, torch.Tensor):
        audio_array = waveform.cpu().numpy()
    else:
        audio_array = np.array(waveform)
    
    # Ensure mono audio
    if audio_array.ndim > 1:
        if audio_array.shape[0] == 2:  # Stereo (channels, samples)
            audio_array = np.mean(audio_array, axis=0)
        elif audio_array.shape[1] == 2:  # Stereo (samples, channels)
            audio_array = np.mean(audio_array, axis=1)
    
    # Ensure proper range [-1, 1] for soundfile
    if audio_array.dtype == np.int16:
        audio_array = audio_array.astype(np.float32) / 32768.0
    elif audio_array.dtype == np.int32:
        audio_array = audio_array.astype(np.float32) / 2147483648.0
    elif audio_array.max() > 1.0:
        audio_array = audio_array / audio_array.max()
    
    # Write to file
    sf.write(output_path, audio_array, sample_rate)

def video_file_to_frames(video_path: str) -> List[np.ndarray]:
    """
    Convert video file back to list of frame arrays
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    finally:
        cap.release()
    
    if len(frames) == 0:
        raise ValueError("No frames extracted from video")
    
    return frames

def check_ffmpeg_availability():
    """
    Check if FFmpeg is available in the system PATH
    """
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            logging.info("FFmpeg is available")
            return True
        else:
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def check_ffprobe_availability():
    """
    Check if FFprobe is available in the system PATH
    """
    try:
        result = subprocess.run(
            ['ffprobe', '-version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
