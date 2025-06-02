import cv2
import numpy as np
import torch
import os
import tempfile
from pathlib import Path

def tensor_to_cv2(tensor):
    """Convert PyTorch tensor to OpenCV image"""
    if len(tensor.shape) == 4:
        # Batch of images
        return [tensor_to_cv2(tensor[i]) for i in range(tensor.shape[0])]
    
    # Single image
    img = tensor.cpu().numpy()
    
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    
    if len(img.shape) == 3 and img.shape[0] in [1, 3, 4]:
        # Channel first to channel last
        img = np.transpose(img, (1, 2, 0))
    
    if img.shape[-1] == 3:
        # RGB to BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img

def cv2_to_tensor(img):
    """Convert OpenCV image to PyTorch tensor"""
    if isinstance(img, list):
        # List of images
        return torch.stack([cv2_to_tensor(im) for im in img])
    
    if img.shape[-1] == 3:
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize to 0-1
    img = img.astype(np.float32) / 255.0
    
    # Convert to tensor and move channel to first dimension
    tensor = torch.from_numpy(img)
    if len(tensor.shape) == 3:
        tensor = tensor.permute(2, 0, 1)
    
    return tensor

def save_temp_video(frames, fps=25.0, format='mp4'):
    """Save frames as temporary video file"""
    temp_file = tempfile.NamedTemporaryFile(suffix=f'.{format}', delete=False)
    temp_path = temp_file.name
    temp_file.close()
    
    if not frames:
        raise ValueError("No frames to save")
    
    # Convert frames to cv2 format if needed
    if isinstance(frames, torch.Tensor):
        frames = tensor_to_cv2(frames)
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    
    for frame in frames:
        writer.write(frame)
    
    writer.release()
    return temp_path

def load_temp_video(video_path):
    """Load video file as frames"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    # Clean up temp file if it exists
    if video_path and os.path.exists(video_path):
        try:
            os.unlink(video_path)
        except:
            pass
    
    return frames

def ensure_audio_format(audio_tensor, target_sr=22050):
    """Ensure audio tensor is in the correct format"""
    import torchaudio
    
    if audio_tensor is None:
        return torch.zeros(1, target_sr)  # 1 second of silence
    
    # Ensure it's a tensor
    if isinstance(audio_tensor, np.ndarray):
        audio_tensor = torch.from_numpy(audio_tensor)
    
    # Ensure it's 2D (channels, samples)
    if len(audio_tensor.shape) == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    elif len(audio_tensor.shape) == 3:
        audio_tensor = audio_tensor.squeeze(0)
    
    return audio_tensor

def get_video_info(video_path):
    """Get basic video information"""
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    return {
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'duration': frame_count / fps if fps > 0 else 0
    }

def validate_inputs(source_video, target_video, audio):
    """Validate input tensors"""
    errors = []
    
    if source_video is None:
        errors.append("Source video is required")
    elif len(source_video.shape) != 4:
        errors.append("Source video must be 4D tensor (batch, height, width, channels)")
    
    if target_video is None:
        errors.append("Target video is required")  
    elif len(target_video.shape) != 4:
        errors.append("Target video must be 4D tensor (batch, height, width, channels)")
    
    if audio is None:
        errors.append("Audio is required")
    elif len(audio.shape) not in [1, 2]:
        errors.append("Audio must be 1D or 2D tensor")
    
    if source_video is not None and target_video is not None:
        if source_video.shape[0] != target_video.shape[0]:
            errors.append("Source and target videos must have same number of frames")
    
    if errors:
        raise ValueError(f"Input validation failed: {'; '.join(errors)}")
    
    return True

def setup_paths():
    """Setup necessary paths for models and temp files"""
    # Get ComfyUI models directory
    import folder_paths
    
    models_dir = folder_paths.models_dir
    keysync_models_dir = os.path.join(models_dir, "keysync")
    
    os.makedirs(keysync_models_dir, exist_ok=True)
    
    return {
        'models_dir': models_dir,
        'keysync_models_dir': keysync_models_dir,
    }
