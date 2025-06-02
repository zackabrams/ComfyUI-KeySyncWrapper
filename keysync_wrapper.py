import os
import sys
import torch
import numpy as np
import cv2
from pathlib import Path
import subprocess
import tempfile
import shutil

# Add KeySync to path - this assumes KeySync is installed or cloned
try:
    from keysync import KeySync
    KEYSYNC_AVAILABLE = True
except ImportError:
    print("KeySync not found. Installing...")
    KEYSYNC_AVAILABLE = False


class KeySyncProcessor:
    """
    Wrapper class for KeySync processing
    """
    
    def __init__(self, device="auto", model_path=None):
        self.device = self._get_device(device)
        self.model = None
        self.model_path = model_path
        self._ensure_keysync()
        self._load_model()
    
    def _get_device(self, device):
        """Determine the best device to use"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _ensure_keysync(self):
        """Ensure KeySync is available"""
        global KEYSYNC_AVAILABLE
        
        if not KEYSYNC_AVAILABLE:
            try:
                self._install_keysync()
                from keysync import KeySync
                KEYSYNC_AVAILABLE = True
            except Exception as e:
                raise ImportError(f"Failed to install or import KeySync: {e}")
    
    def _install_keysync(self):
        """Install KeySync if not available"""
        import subprocess
        import sys
        
        # Clone KeySync repository
        keysync_dir = os.path.join(os.path.dirname(__file__), "keysync")
        
        if not os.path.exists(keysync_dir):
            print("Cloning KeySync repository...")
            subprocess.run([
                "git", "clone", 
                "https://github.com/antonibigata/keysync.git",
                keysync_dir
            ], check=True)
        
        # Add to Python path
        if keysync_dir not in sys.path:
            sys.path.insert(0, keysync_dir)
        
        # Install requirements
        requirements_file = os.path.join(keysync_dir, "requirements.txt")
        if os.path.exists(requirements_file):
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "-r", requirements_file
            ], check=True)
    
    def _load_model(self):
        """Load the KeySync model"""
        try:
            # This is a placeholder - adapt to actual KeySync model loading
            from keysync import KeySync
            self.model = KeySync(device=self.device)
            print(f"KeySync model loaded on {self.device}")
        except Exception as e:
            print(f"Failed to load KeySync model: {e}")
            self.model = None
    
    def process(self, source_video, target_video, audio, output_path, 
                face_enhance=True, face_enhance_model="gfpgan", 
                batch_size=8, quality="medium", mask=None, **kwargs):
        """
        Process video with KeySync
        
        Args:
            source_video: Path to source video
            target_video: Path to target video  
            audio: Path to audio file
            output_path: Path for output video
            face_enhance: Whether to enhance faces
            face_enhance_model: Model for face enhancement
            batch_size: Batch size for processing
            quality: Quality setting
            mask: Optional mask
            **kwargs: Additional parameters
            
        Returns:
            Path to output video
        """
        
        if self.model is None:
            raise RuntimeError("KeySync model not loaded")
        
        try:
            # Prepare arguments for KeySync
            args = {
                'source': source_video,
                'target': target_video, 
                'audio': audio,
                'output': output_path,
                'device': self.device,
                'batch_size': batch_size,
                'enhance': face_enhance,
                'enhancer': face_enhance_model if face_enhance else None,
                'quality': quality,
            }
            
            # Add additional kwargs
            args.update(kwargs)
            
            # Process with KeySync - adapt this to actual KeySync API
            result = self.model.process(**args)
            
            if os.path.exists(output_path):
                return output_path
            else:
                raise RuntimeError("KeySync processing failed - no output generated")
                
        except Exception as e:
            print(f"KeySync processing error: {e}")
            raise e
    
    def process_frames(self, source_frames, target_frames, audio_array, 
                      fps=25.0, **kwargs):
        """
        Process frames directly without file I/O
        
        Args:
            source_frames: List of source frames (numpy arrays)
            target_frames: List of target frames (numpy arrays)  
            audio_array: Audio data as numpy array
            fps: Frames per second
            **kwargs: Additional parameters
            
        Returns:
            List of processed frames, processed audio
        """
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save temporary files
            source_path = os.path.join(temp_dir, "source.mp4")
            target_path = os.path.join(temp_dir, "target.mp4")
            audio_path = os.path.join(temp_dir, "audio.wav")
            output_path = os.path.join(temp_dir, "output.mp4")
            
            # Convert frames to video
            self._frames_to_video(source_frames, source_path, fps)
            self._frames_to_video(target_frames, target_path, fps)
            self._array_to_audio(audio_array, audio_path)
            
            # Process
            result_path = self.process(
                source_path, target_path, audio_path, output_path, **kwargs
            )
            
            # Load result
            result_frames = self._video_to_frames(result_path)
            result_audio = self._audio_to_array(result_path)
            
            return result_frames, result_audio
    
    def _frames_to_video(self, frames, output_path, fps):
        """Convert frame list to video file"""
        if not frames:
            raise ValueError("No frames provided")
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            writer.write(frame)
        
        writer.release()
    
    def _array_to_audio(self, audio_array, output_path, sample_rate=22050):
        """Convert audio array to audio file"""
        import torchaudio
        
        if isinstance(audio_array, np.ndarray):
            audio_tensor = torch.from_numpy(audio_array)
        else:
            audio_tensor = audio_array
        
        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        torchaudio.save(output_path, audio_tensor, sample_rate)
    
    def _video_to_frames(self, video_path):
        """Convert video file to frame list"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        return frames
    
    def _audio_to_array(self, video_path):
        """Extract audio from video as array"""
        import torchaudio
        
        try:
            audio, sample_rate = torchaudio.load(video_path)
            return audio
        except Exception as e:
            print(f"Failed to extract audio: {e}")
            return torch.zeros(1, 22050)  # 1 second of silence
