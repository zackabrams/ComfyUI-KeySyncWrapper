import torch
import numpy as np
import cv2
import os
import tempfile
import folder_paths
from pathlib import Path
import subprocess
import sys

from .keysync_wrapper import KeySyncProcessor
from .utils import tensor_to_cv2, cv2_to_tensor, save_temp_video, load_temp_video

class KeySyncVideoNode:
    """
    Main KeySync node for video-to-video lip sync generation
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_video": ("IMAGE",),  # Source video frames
                "target_video": ("IMAGE",),   # Target video frames with desired speech
                "audio": ("AUDIO",),          # Audio for lip sync
                "face_enhance": ("BOOLEAN", {"default": True}),
                "face_enhance_model": (["gfpgan", "codeformer", "none"], {"default": "gfpgan"}),
                "batch_size": ("INT", {"default": 8, "min": 1, "max": 32}),
                "quality": (["low", "medium", "high"], {"default": "medium"}),
            },
            "optional": {
                "mask": ("MASK",),
                "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 60.0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("synced_video", "audio")
    FUNCTION = "process_keysync"
    CATEGORY = "KeySync"
    
    def __init__(self):
        self.keysync_processor = None
    
    def process_keysync(self, source_video, target_video, audio, face_enhance, 
                       face_enhance_model, batch_size, quality, mask=None, fps=25.0):
        
        # Initialize processor if needed
        if self.keysync_processor is None:
            self.keysync_processor = KeySyncProcessor()
        
        try:
            # Convert ComfyUI tensors to appropriate formats
            source_frames = self._tensor_to_frames(source_video)
            target_frames = self._tensor_to_frames(target_video)
            
            # Save temporary files
            with tempfile.TemporaryDirectory() as temp_dir:
                source_video_path = os.path.join(temp_dir, "source.mp4")
                target_video_path = os.path.join(temp_dir, "target.mp4")
                audio_path = os.path.join(temp_dir, "audio.wav")
                output_path = os.path.join(temp_dir, "output.mp4")
                
                # Save videos and audio
                self._save_video(source_frames, source_video_path, fps)
                self._save_video(target_frames, target_video_path, fps)
                self._save_audio(audio, audio_path)
                
                # Process with KeySync
                result_path = self.keysync_processor.process(
                    source_video=source_video_path,
                    target_video=target_video_path,
                    audio=audio_path,
                    output_path=output_path,
                    face_enhance=face_enhance,
                    face_enhance_model=face_enhance_model,
                    batch_size=batch_size,
                    quality=quality,
                    mask=mask
                )
                
                # Load result back
                result_frames, result_audio = self._load_result(result_path)
                
                return (result_frames, result_audio)
                
        except Exception as e:
            print(f"KeySync processing error: {str(e)}")
            raise e
    
    def _tensor_to_frames(self, tensor):
        """Convert ComfyUI image tensor to list of cv2 frames"""
        frames = []
        for i in range(tensor.shape[0]):
            frame = tensor[i].cpu().numpy()
            frame = (frame * 255).astype(np.uint8)
            if frame.shape[-1] == 3:  # RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frames.append(frame)
        return frames
    
    def _save_video(self, frames, path, fps):
        """Save frames as video file"""
        if not frames:
            raise ValueError("No frames to save")
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        
        for frame in frames:
            writer.write(frame)
        writer.release()
    
    def _save_audio(self, audio_tensor, path):
        """Save audio tensor to file"""
        # Assuming audio is in format [samples, channels] or [samples]
        import torchaudio
        
        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Save as WAV
        torchaudio.save(path, audio_tensor, 22050)
    
    def _load_result(self, video_path):
        """Load processed video back to tensors"""
        import torchaudio
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        
        cap.release()
        
        if not frames:
            raise ValueError("No frames loaded from result video")
        
        # Convert to tensor
        frames_tensor = torch.from_numpy(np.array(frames))
        
        # Load audio (extract from video or use original)
        try:
            audio_tensor, _ = torchaudio.load(video_path)
        except:
            # Fallback - create silent audio
            audio_tensor = torch.zeros(1, len(frames) * 1000)  # Approximate
        
        return frames_tensor, audio_tensor


class LoadKeySync:
    """
    Node to load and initialize KeySync model
    """
    
    @classmethod  
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (["keysync"], {"default": "keysync"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            }
        }
    
    RETURN_TYPES = ("KEYSYNC_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "KeySync"
    
    def load_model(self, model_name, device):
        processor = KeySyncProcessor(device=device)
        return (processor,)


class KeySyncAdvancedNode:
    """
    Advanced KeySync node with more options
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("KEYSYNC_MODEL",),
                "source_video": ("IMAGE",),
                "target_video": ("IMAGE",),  
                "audio": ("AUDIO",),
                "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 60.0}),
            },
            "optional": {
                "face_enhance": ("BOOLEAN", {"default": True}),
                "face_enhance_model": (["gfpgan", "codeformer", "none"], {"default": "gfpgan"}),
                "batch_size": ("INT", {"default": 8, "min": 1, "max": 32}),
                "quality": (["low", "medium", "high"], {"default": "medium"}),
                "crop_face": ("BOOLEAN", {"default": True}),
                "smooth_face": ("BOOLEAN", {"default": True}),
                "mask": ("MASK",),
                "face_detection_confidence": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("synced_video", "audio") 
    FUNCTION = "process_advanced"
    CATEGORY = "KeySync"
    
    def process_advanced(self, model, source_video, target_video, audio, fps, **kwargs):
        # Similar implementation to KeySyncVideoNode but using pre-loaded model
        # and additional advanced options
        
        try:
            source_frames = self._tensor_to_frames(source_video)
            target_frames = self._tensor_to_frames(target_video)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                source_video_path = os.path.join(temp_dir, "source.mp4")
                target_video_path = os.path.join(temp_dir, "target.mp4")
                audio_path = os.path.join(temp_dir, "audio.wav")
                output_path = os.path.join(temp_dir, "output.mp4")
                
                self._save_video(source_frames, source_video_path, fps)
                self._save_video(target_frames, target_video_path, fps)
                self._save_audio(audio, audio_path)
                
                result_path = model.process(
                    source_video=source_video_path,
                    target_video=target_video_path,
                    audio=audio_path,
                    output_path=output_path,
                    **kwargs
                )
                
                result_frames, result_audio = self._load_result(result_path)
                return (result_frames, result_audio)
                
        except Exception as e:
            print(f"KeySync advanced processing error: {str(e)}")
            raise e
    
    def _tensor_to_frames(self, tensor):
        """Convert ComfyUI image tensor to list of cv2 frames"""
        frames = []
        for i in range(tensor.shape[0]):
            frame = tensor[i].cpu().numpy()
            frame = (frame * 255).astype(np.uint8)
            if frame.shape[-1] == 3:  # RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frames.append(frame)
        return frames
    
    def _save_video(self, frames, path, fps):
        """Save frames as video file"""
        if not frames:
            raise ValueError("No frames to save")
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        
        for frame in frames:
            writer.write(frame)
        writer.release()
    
    def _save_audio(self, audio_tensor, path):
        """Save audio tensor to file"""
        import torchaudio
        
        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        torchaudio.save(path, audio_tensor, 22050)
    
    def _load_result(self, video_path):
        """Load processed video back to tensors"""
        import torchaudio
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        
        cap.release()
        
        if not frames:
            raise ValueError("No frames loaded from result video")
        
        frames_tensor = torch.from_numpy(np.array(frames))
        
        try:
            audio_tensor, _ = torchaudio.load(video_path)
        except:
            audio_tensor = torch.zeros(1, len(frames) * 1000)
        
        return frames_tensor, audio_tensor


# Node mappings
NODE_CLASS_MAPPINGS = {
    "KeySyncVideo": KeySyncVideoNode,
    "LoadKeySync": LoadKeySync,
    "KeySyncAdvanced": KeySyncAdvancedNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KeySyncVideo": "KeySync Video Lip Sync",
    "LoadKeySync": "Load KeySync Model", 
    "KeySyncAdvanced": "KeySync Advanced",
}
