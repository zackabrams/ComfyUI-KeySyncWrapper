#!/usr/bin/env python3
"""
KeySync Wrapper - Simple and Clean ComfyUI Integration
Directly wraps KeySync's dubbing_pipeline_raw.py without unnecessary complexity.
"""

import os
import sys
import uuid
import shutil
import tempfile
import atexit

# ----------------------------
# Temp directory management (essential parts from LatentSync)
# ----------------------------

def get_comfyui_temp_dir():
    """Dynamically locate ComfyUI's temp directory."""
    try:
        import folder_paths
        comfy_root = os.path.dirname(os.path.dirname(os.path.abspath(folder_paths.__file__)))
        temp_dir = os.path.join(comfy_root, "temp")
        return temp_dir
    except:
        pass
    
    try:
        current = os.path.dirname(os.path.abspath(__file__))
        for _ in range(5):
            if os.path.exists(os.path.join(current, "comfy.py")):
                return os.path.join(current, "temp")
            current = os.path.dirname(current)
    except:
        pass
    
    return None

def cleanup_comfyui_temp_directories():
    """Clean up ComfyUI temp directories to avoid conflicts."""
    comfyui_temp = get_comfyui_temp_dir()
    if not comfyui_temp:
        return
    
    comfyui_base = os.path.dirname(comfyui_temp)
    
    if os.path.exists(comfyui_temp):
        try:
            shutil.rmtree(comfyui_temp)
        except:
            try:
                backup = f"{comfyui_temp}_backup_{uuid.uuid4().hex[:8]}"
                os.rename(comfyui_temp, backup)
            except:
                pass
    
    try:
        for name in os.listdir(comfyui_base):
            if name.startswith("temp_backup_"):
                path = os.path.join(comfyui_base, name)
                shutil.rmtree(path, ignore_errors=True)
    except:
        pass

def init_temp_directories():
    """Create isolated temp directory for KeySync."""
    cleanup_comfyui_temp_directories()
    
    system_temp = tempfile.gettempdir()
    unique_id = str(uuid.uuid4())[:8]
    module_temp = os.path.join(system_temp, f"keysync_{unique_id}")
    os.makedirs(module_temp, exist_ok=True)
    
    # Override environment variables
    os.environ["TMPDIR"] = module_temp
    os.environ["TEMP"] = module_temp
    os.environ["TMP"] = module_temp
    tempfile.tempdir = module_temp
    
    # Clean up any existing ComfyUI temp
    comfyui_temp = get_comfyui_temp_dir()
    if comfyui_temp and os.path.exists(comfyui_temp):
        try:
            shutil.rmtree(comfyui_temp)
        except:
            try:
                backup = f"{comfyui_temp}_backup_{unique_id}"
                os.rename(comfyui_temp, backup)
                shutil.rmtree(backup, ignore_errors=True)
            except:
                pass
    
    print(f"[KeySync] Initialized temp directory: {module_temp}")
    return module_temp

def module_cleanup():
    """Clean up module temp directory on exit."""
    global MODULE_TEMP_DIR
    try:
        if MODULE_TEMP_DIR and os.path.exists(MODULE_TEMP_DIR):
            shutil.rmtree(MODULE_TEMP_DIR, ignore_errors=True)
            print(f"[KeySync] Cleaned up temp directory: {MODULE_TEMP_DIR}")
    except:
        pass
    cleanup_comfyui_temp_directories()

# Initialize temp management
MODULE_TEMP_DIR = init_temp_directories()
atexit.register(module_cleanup)

# Override folder_paths if available
try:
    import folder_paths
    folder_paths.get_temp_directory = lambda *args, **kwargs: MODULE_TEMP_DIR
except:
    pass

# ----------------------------
# Regular imports
# ----------------------------
import torch
import numpy as np
from PIL import Image
import soundfile as sf


# ----------------------------
# Lazy loading of inference module
# ----------------------------
infer = None

def _get_infer_module():
    """Lazily load infer_simple.py when needed."""
    global infer
    if infer is None:
        import importlib.util
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "infer_simple.py")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Could not find infer_simple.py at {path}")
        spec = importlib.util.spec_from_file_location("keysync_infer_simple", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        infer = module
    return infer

# ----------------------------
# KeySyncWrapper Node - Simple and Clean
# ----------------------------
class KeySyncWrapper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "KeySync"

    def process(self, frames, audio):
        """
        Simple wrapper: Convert ComfyUI tensors → KeySync files → ComfyUI tensors
        """
        # Create run-specific directory
        run_id = str(uuid.uuid4())[:8]
        run_dir = os.path.join(MODULE_TEMP_DIR, f"run_{run_id}")
        os.makedirs(run_dir, exist_ok=True)

        try:
            # Convert frames to numpy array
            if isinstance(frames, torch.Tensor):
                frames_cpu = frames.cpu().numpy()  # [N,H,W,3] float32 in [0..1]
                frames_uint8 = (frames_cpu * 255.0).clip(0, 255).astype("uint8")
            else:
                # Handle list of PIL Images
                arrs = []
                for img in frames:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    arr = np.array(img).astype("uint8")
                    arrs.append(arr)
                frames_uint8 = np.stack(arrs, axis=0)

            # Create KeySync directory structure
            videos_dir = os.path.join(run_dir, "videos")
            audios_dir = os.path.join(run_dir, "audios")
            os.makedirs(videos_dir, exist_ok=True)
            os.makedirs(audios_dir, exist_ok=True)

            # Save video file in videos directory
            temp_video_path = os.path.join(videos_dir, "input.mp4")
            self._save_video(frames_uint8, temp_video_path)

            # Save audio file in audios directory  
            temp_audio_path = os.path.join(audios_dir, "input.wav")
            self._save_audio(audio, temp_audio_path)

            # Create filelist files with absolute paths for reliable path resolution
            # This bypasses KeySync's path replacement logic and works from any working directory
            video_filelist = os.path.join(run_dir, "video_filelist.txt")
            audio_filelist = os.path.join(run_dir, "audio_filelist.txt")
            
            # Write absolute paths to the filelist files for reliable path resolution
            with open(video_filelist, 'w') as f:
                f.write(os.path.abspath(temp_video_path) + '\n')
            
            with open(audio_filelist, 'w') as f:
                f.write(os.path.abspath(temp_audio_path) + '\n')

            # Prepare output directory - KeySync will create subdirectories under this
            output_base_dir = os.path.join(run_dir, "output")
            os.makedirs(output_base_dir, exist_ok=True)

            # Check for model files
            base_dir = os.path.dirname(os.path.realpath(__file__))
            model_dir = os.path.join(base_dir, "pretrained_models")
            keyframe_ckpt = os.path.join(model_dir, "keyframe_dub.pt")
            interpolation_ckpt = os.path.join(model_dir, "interpolation_dub.pt")
            
            if not os.path.isfile(keyframe_ckpt) or not os.path.isfile(interpolation_ckpt):
                error_msg = (
                    "KeySync models not found! Please download:\n"
                    "git lfs install && git clone https://huggingface.co/toninio19/keysync pretrained_models"
                )
                print(f"[KeySync Error] {error_msg}")
                return (frames,)

            # Calculate processing duration
            num_frames = frames_uint8.shape[0]
            fps = 25
            compute_secs = max(int(num_frames / fps), 1)

            print(f"[KeySync] Processing {num_frames} frames ({compute_secs}s) through KeySync...")
            print(f"[KeySync] Video shape: {frames_uint8.shape}")
            print(f"[KeySync] Audio shape: {audio['waveform'].shape} at {audio.get('sample_rate', 16000)}Hz")

            # Run KeySync inference
            infer_module = _get_infer_module()
            infer_module.run_keysync_inference(
                video_dir=run_dir,  # Directory containing the filelist files
                audio_dir=run_dir,  # Directory containing the filelist files
                output_dir=output_base_dir,  # Base output directory
                keyframe_ckpt=keyframe_ckpt,
                interpolation_ckpt=interpolation_ckpt,
                compute_until=compute_secs,
                fix_occlusion=False,
                position=None,
                start_frame=0
            )

            # Load output frames - check multiple possible locations
            # Wait a moment for files to be created
            import time
            time.sleep(1.0)
            
            # Check for frames in the expected synced_frames directory
            output_frame_dir = os.path.join(output_base_dir, "synced_frames")
            png_files = []
            
            if os.path.exists(output_frame_dir):
                png_files = sorted([
                    f for f in os.listdir(output_frame_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ])
                if png_files:
                    print(f"[KeySync] Found {len(png_files)} frames in synced_frames directory")
            
            # If no frames in synced_frames, check the base output directory
            if len(png_files) == 0 and os.path.exists(output_base_dir):
                png_files = sorted([
                    f for f in os.listdir(output_base_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ])
                if png_files:
                    print(f"[KeySync] Found {len(png_files)} frames in base output directory")
                    output_frame_dir = output_base_dir  # Update the directory path
            
            if len(png_files) == 0:
                print(f"[KeySync] No output frames found in {output_base_dir}")
                print(f"[KeySync] Returning original frames")
                return (frames,)

            # Convert back to ComfyUI tensor format
            loaded_list = []
            for fname in png_files:
                img_path = os.path.join(output_frame_dir, fname)
                img = Image.open(img_path).convert("RGB")
                arr = np.array(img).astype("float32") / 255.0  # [H,W,3] float32
                tensor_frame = torch.from_numpy(arr)
                loaded_list.append(tensor_frame)
                img.close()

            synced_tensor = torch.stack(loaded_list, dim=0)  # [N,H,W,3]
            print(f"[KeySync] Successfully processed {len(loaded_list)} frames")
            
            return (synced_tensor,)

        except Exception as e:
            print(f"[KeySync Error] {e}")
            import traceback
            traceback.print_exc()
            return (frames,)

        finally:
            # Clean up run directory
            if run_dir and os.path.exists(run_dir):
                try:
                    shutil.rmtree(run_dir, ignore_errors=True)
                except Exception as cleanup_error:
                    print(f"[KeySync] Cleanup warning: {cleanup_error}")

    def _save_video(self, frames, video_path):
        """Save frames as MP4 with fallback options."""
        try:
            # Try torchvision first
            import torchvision.io as io
            frames_tensor = torch.from_numpy(frames)
            io.write_video(video_path, frames_tensor, fps=25, video_codec="h264")
            return
        except TypeError as e:
            if "macro_block_size" in str(e):
                try:
                    import imageio
                    imageio.mimsave(video_path, frames, fps=25, codec="h264", macro_block_size=1)
                    return
                except Exception:
                    pass
        except Exception:
            pass
        
        # Try imageio
        try:
            import imageio
            imageio.mimsave(video_path, frames, fps=25, codec="h264")
            return
        except Exception as e:
            print(f"[KeySync] Video save failed, using PNG sequence fallback: {e}")
            # Fallback: save as PNG sequence
            for i, frame in enumerate(frames):
                frame_path = os.path.join(os.path.dirname(video_path), f"frame_{i:04d}.png")
                Image.fromarray(frame).save(frame_path)

    def _save_audio(self, audio, audio_path):
        """Save audio in format expected by KeySync."""
        waveform = audio.get("waveform", None)
        sr = int(audio.get("sample_rate", 16000))
        
        if waveform is None:
            raise RuntimeError("No waveform in audio dict")
        
        # Convert to numpy and ensure mono
        wav_tensor = waveform.squeeze().cpu().numpy()
        if wav_tensor.ndim > 1:
            wav_tensor = wav_tensor.mean(axis=0)
        
        # Ensure proper range
        wav_tensor = wav_tensor.astype(np.float32)
        if wav_tensor.max() > 1.0 or wav_tensor.min() < -1.0:
            wav_tensor = wav_tensor / max(abs(wav_tensor.max()), abs(wav_tensor.min()))
        
        # KeySync expects 16kHz audio
        if sr != 16000:
            print(f"[KeySync] Resampling audio from {sr}Hz to 16000Hz")
            import torchaudio
            # Convert back to tensor for resampling
            wav_torch = torch.from_numpy(wav_tensor).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            wav_torch = resampler(wav_torch)
            wav_tensor = wav_torch.squeeze().numpy()
            sr = 16000
        
        sf.write(audio_path, wav_tensor, sr)
        print(f"[KeySync] Saved audio: {audio_path} (shape: {wav_tensor.shape}, sr: {sr})")


# ----------------------------
# KeySyncOptimized Node - For A100-80GB High-VRAM GPUs
# ----------------------------
class KeySyncOptimized:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "audio": ("AUDIO",),
                "decoding_batch_size": ("INT", {"default": 8, "min": 1, "max": 32, "step": 1}),
                "chunk_size": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1}),
                "vae_batch_size": ("INT", {"default": 16, "min": 1, "max": 64, "step": 1}),
            },
            "optional": {
                "compute_until": ("INT", {"default": 45, "min": 1, "max": 300, "step": 1}),
                "fix_occlusion": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "KeySync"

    def process(self, frames, audio, decoding_batch_size=8, chunk_size=4, vae_batch_size=16, 
                compute_until=45, fix_occlusion=False):
        """
        Optimized KeySync wrapper for high-VRAM GPUs with exposed batch size parameters.
        """
        # Create run-specific directory
        run_id = str(uuid.uuid4())[:8]
        run_dir = os.path.join(MODULE_TEMP_DIR, f"run_{run_id}")
        os.makedirs(run_dir, exist_ok=True)

        try:
            # Convert frames to numpy array
            if isinstance(frames, torch.Tensor):
                frames_cpu = frames.cpu().numpy()  # [N,H,W,3] float32 in [0..1]
                frames_uint8 = (frames_cpu * 255.0).clip(0, 255).astype("uint8")
            else:
                # Handle list of PIL Images
                arrs = []
                for img in frames:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    arr = np.array(img).astype("uint8")
                    arrs.append(arr)
                frames_uint8 = np.stack(arrs, axis=0)

            # Create KeySync directory structure
            videos_dir = os.path.join(run_dir, "videos")
            audios_dir = os.path.join(run_dir, "audios")
            os.makedirs(videos_dir, exist_ok=True)
            os.makedirs(audios_dir, exist_ok=True)

            # Save video file in videos directory
            temp_video_path = os.path.join(videos_dir, "input.mp4")
            self._save_video(frames_uint8, temp_video_path)

            # Save audio file in audios directory  
            temp_audio_path = os.path.join(audios_dir, "input.wav")
            self._save_audio(audio, temp_audio_path)

            # Create filelist files with absolute paths for reliable path resolution
            # This bypasses KeySync's path replacement logic and works from any working directory
            video_filelist = os.path.join(run_dir, "video_filelist.txt")
            audio_filelist = os.path.join(run_dir, "audio_filelist.txt")
            
            # Write absolute paths to the filelist files for reliable path resolution
            with open(video_filelist, 'w') as f:
                f.write(os.path.abspath(temp_video_path) + '\n')
            
            with open(audio_filelist, 'w') as f:
                f.write(os.path.abspath(temp_audio_path) + '\n')

            # Prepare output directory - KeySync will create subdirectories under this
            output_base_dir = os.path.join(run_dir, "output")
            os.makedirs(output_base_dir, exist_ok=True)

            # Check for model files
            base_dir = os.path.dirname(os.path.realpath(__file__))
            model_dir = os.path.join(base_dir, "pretrained_models")
            keyframe_ckpt = os.path.join(model_dir, "keyframe_dub.pt")
            interpolation_ckpt = os.path.join(model_dir, "interpolation_dub.pt")
            
            if not os.path.isfile(keyframe_ckpt) or not os.path.isfile(interpolation_ckpt):
                error_msg = (
                    "KeySync models not found! Please download:\n"
                    "git lfs install && git clone https://huggingface.co/toninio19/keysync pretrained_models"
                )
                print(f"[KeySync Error] {error_msg}")
                return (frames,)

            # Calculate processing duration
            num_frames = frames_uint8.shape[0]
            fps = 25
            compute_secs = max(int(num_frames / fps), 1)
            if compute_until and compute_until > 0:
                compute_secs = min(compute_secs, compute_until)

            print(f"[KeySync Optimized] Processing {num_frames} frames ({compute_secs}s) with optimized settings:")
            print(f"[KeySync Optimized] - Decoding batch size: {decoding_batch_size}")
            print(f"[KeySync Optimized] - Chunk size: {chunk_size}")
            print(f"[KeySync Optimized] - VAE batch size: {vae_batch_size}")
            print(f"[KeySync Optimized] Video shape: {frames_uint8.shape}")
            print(f"[KeySync Optimized] Audio shape: {audio['waveform'].shape} at {audio.get('sample_rate', 16000)}Hz")

            # Run optimized KeySync inference
            infer_module = _get_infer_module()
            infer_module.run_keysync_inference_optimized(
                video_dir=run_dir,  # Directory containing the filelist files
                audio_dir=run_dir,  # Directory containing the filelist files
                output_dir=output_base_dir,  # Base output directory
                keyframe_ckpt=keyframe_ckpt,
                interpolation_ckpt=interpolation_ckpt,
                compute_until=compute_secs,
                fix_occlusion=fix_occlusion,
                position=None,
                start_frame=0,
                decoding_batch_size=decoding_batch_size,
                chunk_size=chunk_size,
                vae_batch_size=vae_batch_size
            )

            # Load output frames - check multiple possible locations
            # Wait a moment for files to be created
            import time
            time.sleep(1.0)
            
            # Check for frames in the expected synced_frames directory
            output_frame_dir = os.path.join(output_base_dir, "synced_frames")
            png_files = []
            
            if os.path.exists(output_frame_dir):
                png_files = sorted([
                    f for f in os.listdir(output_frame_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ])
                if png_files:
                    print(f"[KeySync Optimized] Found {len(png_files)} frames in synced_frames directory")
            
            # If no frames in synced_frames, check the base output directory
            if len(png_files) == 0 and os.path.exists(output_base_dir):
                png_files = sorted([
                    f for f in os.listdir(output_base_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ])
                if png_files:
                    print(f"[KeySync Optimized] Found {len(png_files)} frames in base output directory")
                    output_frame_dir = output_base_dir  # Update the directory path
            
            if len(png_files) == 0:
                print(f"[KeySync Optimized] No output frames found in {output_base_dir}")
                print(f"[KeySync Optimized] Returning original frames")
                return (frames,)

            # Convert back to ComfyUI tensor format
            loaded_list = []
            for fname in png_files:
                img_path = os.path.join(output_frame_dir, fname)
                img = Image.open(img_path).convert("RGB")
                arr = np.array(img).astype("float32") / 255.0  # [H,W,3] float32
                tensor_frame = torch.from_numpy(arr)
                loaded_list.append(tensor_frame)
                img.close()

            synced_tensor = torch.stack(loaded_list, dim=0)  # [N,H,W,3]
            print(f"[KeySync Optimized] Successfully processed {len(loaded_list)} frames with optimized settings")
            
            return (synced_tensor,)

        except Exception as e:
            print(f"[KeySync Optimized Error] {e}")
            import traceback
            traceback.print_exc()
            return (frames,)

        finally:
            # Clean up run directory
            if run_dir and os.path.exists(run_dir):
                try:
                    shutil.rmtree(run_dir, ignore_errors=True)
                except Exception as cleanup_error:
                    print(f"[KeySync Optimized] Cleanup warning: {cleanup_error}")

    def _save_video(self, frames, video_path):
        """Save frames as MP4 with fallback options."""
        try:
            # Try torchvision first
            import torchvision.io as io
            frames_tensor = torch.from_numpy(frames)
            io.write_video(video_path, frames_tensor, fps=25, video_codec="h264")
            return
        except TypeError as e:
            if "macro_block_size" in str(e):
                try:
                    import imageio
                    imageio.mimsave(video_path, frames, fps=25, codec="h264", macro_block_size=1)
                    return
                except Exception:
                    pass
        except Exception:
            pass
        
        # Try imageio
        try:
            import imageio
            imageio.mimsave(video_path, frames, fps=25, codec="h264")
            return
        except Exception as e:
            print(f"[KeySync Optimized] Video save failed, using PNG sequence fallback: {e}")
            # Fallback: save as PNG sequence
            for i, frame in enumerate(frames):
                frame_path = os.path.join(os.path.dirname(video_path), f"frame_{i:04d}.png")
                Image.fromarray(frame).save(frame_path)

    def _save_audio(self, audio, audio_path):
        """Save audio in format expected by KeySync."""
        waveform = audio.get("waveform", None)
        sr = int(audio.get("sample_rate", 16000))
        
        if waveform is None:
            raise RuntimeError("No waveform in audio dict")
        
        # Convert to numpy and ensure mono
        wav_tensor = waveform.squeeze().cpu().numpy()
        if wav_tensor.ndim > 1:
            wav_tensor = wav_tensor.mean(axis=0)
        
        # Ensure proper range
        wav_tensor = wav_tensor.astype(np.float32)
        if wav_tensor.max() > 1.0 or wav_tensor.min() < -1.0:
            wav_tensor = wav_tensor / max(abs(wav_tensor.max()), abs(wav_tensor.min()))
        
        # KeySync expects 16kHz audio
        if sr != 16000:
            print(f"[KeySync Optimized] Resampling audio from {sr}Hz to 16000Hz")
            import torchaudio
            # Convert back to tensor for resampling
            wav_torch = torch.from_numpy(wav_tensor).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            wav_torch = resampler(wav_torch)
            wav_tensor = wav_torch.squeeze().numpy()
            sr = 16000
        
        sf.write(audio_path, wav_tensor, sr)
        print(f"[KeySync Optimized] Saved audio: {audio_path} (shape: {wav_tensor.shape}, sr: {sr})")


# Register nodes
NODE_CLASS_MAPPINGS = {
    "KeySyncWrapper": KeySyncWrapper,
    "KeySyncOptimized": KeySyncOptimized
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KeySyncWrapper": "KeySync Simple",
    "KeySyncOptimized": "KeySync Optimized (A100)"
}
