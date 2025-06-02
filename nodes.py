import torch
import numpy as np
import os
import tempfile
import folder_paths
from .keysync_wrapper import KeySyncCLIWrapper
from .model_downloader import KeySyncModelDownloader
from .preprocessing import prepare_video_for_keysync, prepare_audio_for_keysync
from .utils import frames_to_video_file, video_file_to_frames

class KeySyncSetup:
    """
    Node for setting up KeySync environment and downloading models
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_variant": (["base", "hq"], {"default": "base"}),
                "force_reinstall": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("KEYSYNC_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "setup_keysync"
    CATEGORY = "video/keysync"
    
    def setup_keysync(self, model_variant, force_reinstall=False):
        """Setup KeySync environment and download required models"""
        try:
            downloader = KeySyncModelDownloader()
            config = downloader.setup_keysync(
                variant=model_variant,
                force_reinstall=force_reinstall
            )
            return (config,)
        except Exception as e:
            print(f"KeySync setup error: {str(e)}")
            raise e

class KeySyncNode:
    """
    Main KeySync node using CLI wrapper approach
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "config": ("KEYSYNC_CONFIG",),
                "images": ("IMAGE",),  # Video frames from VideoHelperSuite
                "audio": ("AUDIO",),   # Audio from VideoHelperSuite
            },
            "optional": {
                "fps": ("FLOAT", {
                    "default": 25.0,
                    "min": 1.0,
                    "max": 60.0,
                    "step": 0.1
                }),
                "face_enhance": ("BOOLEAN", {"default": True}),
                "temp_cleanup": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "apply_keysync"
    CATEGORY = "video/keysync"
    
    def apply_keysync(self, config, images, audio, fps=25.0, 
                     face_enhance=True, temp_cleanup=True):
        """
        Apply KeySync lip synchronization using CLI wrapper
        """
        temp_dir = None
        try:
            # Create temporary working directory
            temp_dir = tempfile.mkdtemp(prefix="keysync_")
            
            # Prepare input files for KeySync CLI
            video_path, audio_path = self._prepare_inputs(
                images, audio, fps, temp_dir
            )
            
            # Initialize KeySync wrapper
            wrapper = KeySyncCLIWrapper(config)
            
            # Run KeySync processing
            output_video_path = wrapper.process_lipsync(
                video_path=video_path,
                audio_path=audio_path,
                output_dir=temp_dir,
                face_enhance=face_enhance
            )
            
            # Convert output back to ComfyUI tensor format
            output_frames = self._process_output(output_video_path)
            
            return (output_frames,)
            
        except Exception as e:
            print(f"KeySync processing error: {str(e)}")
            raise e
        finally:
            if temp_cleanup and temp_dir and os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _prepare_inputs(self, images, audio, fps, temp_dir):
        """Prepare video and audio files for KeySync CLI"""
        
        # Convert ComfyUI images tensor to video file
        video_path = os.path.join(temp_dir, "input_video.mp4")
        prepare_video_for_keysync(images, video_path, fps)
        
        # Convert ComfyUI audio to wav file
        audio_path = os.path.join(temp_dir, "input_audio.wav")
        prepare_audio_for_keysync(audio, audio_path)
        
        return video_path, audio_path
    
    def _process_output(self, output_video_path):
        """Convert KeySync output video back to ComfyUI frames"""
        if not os.path.exists(output_video_path):
            raise FileNotFoundError(f"KeySync output not found: {output_video_path}")
        
        # Convert video file back to frame tensors
        frames = video_file_to_frames(output_video_path)
        
        # Convert to ComfyUI tensor format
        if len(frames) == 0:
            raise ValueError("No frames extracted from KeySync output")
        
        # Stack frames and convert to proper format
        stacked_frames = np.stack(frames)
        
        # Ensure correct format: [N, H, W, C] in range [0,1]
        if stacked_frames.dtype == np.uint8:
            stacked_frames = stacked_frames.astype(np.float32) / 255.0
        
        output_tensor = torch.from_numpy(stacked_frames)
        return output_tensor
