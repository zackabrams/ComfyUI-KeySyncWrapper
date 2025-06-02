import os
import sys
import uuid
import shutil
import tempfile
import atexit
import subprocess
from collections.abc import Mapping
from datetime import datetime

# ----------------------------
# Temp‐directory management
# (Copied/adapted from LatentSyncWrapper)
# ----------------------------

def get_comfyui_temp_dir():
    """
    Dynamically locate ComfyUI’s <ComfyUI>/temp directory.
    """
    # 1) Try using folder_paths (if ComfyUI installed it)
    try:
        import folder_paths
        comfy_root = os.path.dirname(os.path.dirname(os.path.abspath(folder_paths.__file__)))
        temp_dir = os.path.join(comfy_root, "temp")
        return temp_dir
    except:
        pass

    # 2) Otherwise, walk up from here until we find comfy.py
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
    """
    Delete any existing ComfyUI temp directory (to avoid conflicts).
    """
    comfyui_temp = get_comfyui_temp_dir()
    if not comfyui_temp:
        return
    comfyui_base = os.path.dirname(comfyui_temp)
    # Remove the main temp folder if it exists
    if os.path.exists(comfyui_temp):
        try:
            shutil.rmtree(comfyui_temp)
        except:
            try:
                backup = f"{comfyui_temp}_backup_{uuid.uuid4().hex[:8]}"
                os.rename(comfyui_temp, backup)
            except:
                pass
    # Also remove any “temp_backup_*” siblings
    try:
        for name in os.listdir(comfyui_base):
            if name.startswith("temp_backup_"):
                path = os.path.join(comfyui_base, name)
                shutil.rmtree(path, ignore_errors=True)
    except:
        pass

def init_temp_directories():
    """
    Create a unique temp directory under the system temp and override
    Python/ComfyUI to use it.
    """
    cleanup_comfyui_temp_directories()
    system_temp = tempfile.gettempdir()
    unique_id = str(uuid.uuid4())[:8]
    module_temp = os.path.join(system_temp, f"keysync_{unique_id}")
    os.makedirs(module_temp, exist_ok=True)

    os.environ["TMPDIR"] = module_temp
    os.environ["TEMP"] = module_temp
    os.environ["TMP"] = module_temp
    tempfile.tempdir = module_temp

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

    return module_temp

def module_cleanup():
    """
    Clean up our module temp dir on exit, and also sweep ComfyUI’s temp.
    """
    global MODULE_TEMP_DIR
    try:
        if MODULE_TEMP_DIR and os.path.exists(MODULE_TEMP_DIR):
            shutil.rmtree(MODULE_TEMP_DIR, ignore_errors=True)
    except:
        pass
    cleanup_comfyui_temp_directories()

# Initialize once at import time
MODULE_TEMP_DIR = init_temp_directories()
atexit.register(module_cleanup)

# Override folder_paths.get_temp_directory() so any ComfyUI code uses our temp
try:
    import folder_paths
    folder_paths.get_temp_directory = lambda *args, **kwargs: MODULE_TEMP_DIR
except:
    pass

# ----------------------------
# Now import normal dependencies
# ----------------------------
import torch
import numpy as np
from PIL import Image
import soundfile as sf

# Import our infer wrapper
import infer

# ----------------------------
# KeySyncWrapper node
# ----------------------------
class KeySyncWrapper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # “frames” comes in as a Tensor [N, H, W, 3], float32 scaled 0..1
                "frames": ("IMAGE",),
                # “audio” is a dict: {"waveform": Tensor[1×T], "sample_rate": int}
                "audio": ("AUDIO",),
                "keypoint_confidence_threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}
                ),
                "sync_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}
                )
            }
        }

    RETURN_TYPES = ("IMAGE",)   # We return a single batched Tensor [N,H,W,3]
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "KeySync"

    def process(self, frames, audio, keypoint_confidence_threshold, sync_strength):
        """
        1. ‘frames’: Tensor [N,H,W,3], dtype=float32 (0..1)
        2. ‘audio’: dict { "waveform": Tensor[1×T], "sample_rate": int }
        3. We write a temp MP4 & WAV, call KeySync’s infer_raw.sh, then read PNGs back.
        4. Return a single Tensor [N,H,W,3] float32 (0..1).
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1) Create a per‐run subdirectory under MODULE_TEMP_DIR
        run_id = str(uuid.uuid4())[:8]
        run_dir = os.path.join(MODULE_TEMP_DIR, f"run_{run_id}")
        os.makedirs(run_dir, exist_ok=True)

        try:
            # ----- Step A: Write input frames to a temporary MP4 ----- #
            # frames is either a Tensor [N,H,W,3] or a list of PIL Images
            # Ensure we get a NumPy array [N,H,W,3], dtype=uint8

            if isinstance(frames, torch.Tensor):
                # Move to CPU & convert to numpy
                frames_cpu = frames.cpu().numpy()  # float32 in [0..1]
                frames_uint8 = (frames_cpu * 255.0).clip(0,255).astype("uint8")
            else:
                # If it came in as a list of PIL Images
                pil_list = frames
                arrs = []
                for img in pil_list:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    arr = np.array(img).astype("uint8")
                    arrs.append(arr)
                frames_uint8 = np.stack(arrs, axis=0)  # [N,H,W,3], uint8

            # Write out temp MP4
            temp_video_path = os.path.join(run_dir, "input.mp4")
            try:
                import torchvision.io as io
                io.write_video(
                    temp_video_path,
                    frames_uint8,
                    fps=25,
                    video_codec="h264"
                )
            except TypeError:
                # Possibly macro_block_size errors; fallback to imageio
                import imageio
                imageio.mimsave(
                    temp_video_path,
                    frames_uint8,
                    fps=25,
                    codec="h264",
                    macro_block_size=1
                )

            # ----- Step B: Write input audio to a WAV (16 kHz) ----- #
            waveform = audio.get("waveform", None)
            sr = int(audio.get("sample_rate", 16000))
            if waveform is None:
                raise RuntimeError("KeySyncWrapper: Received no ‘waveform’ in the AUDIO dict.")
            # waveform is Tensor [1×T] or [T]; move to CPU & numpy
            wav_tensor = waveform.squeeze(0).cpu().numpy()
            temp_audio_path = os.path.join(run_dir, "input.wav")
            sf.write(temp_audio_path, wav_tensor, sr)

            # ----- Step C: Prepare output folder for KeySync’s PNGs ----- #
            output_frame_dir = os.path.join(run_dir, "synced_frames")
            os.makedirs(output_frame_dir, exist_ok=True)

            # ----- Step D: Locate model checkpoints (user must have placed these) ----- #
            base_dir = os.path.dirname(os.path.realpath(__file__))
            model_dir = os.path.join(base_dir, "pretrained_models")
            keyframe_ckpt = os.path.join(model_dir, "keyframe_dub.pt")
            interpolation_ckpt = os.path.join(model_dir, "interpolation_dub.pt")
            if not os.path.isfile(keyframe_ckpt) or not os.path.isfile(interpolation_ckpt):
                raise FileNotFoundError(
                    "KeySyncWrapper: Could not find keyframe_dub.pt and/or interpolation_dub.pt under pretrained_models/"
                )

            # ----- Step E: Compute ‘compute_until’ seconds based on frame count ----- #
            num_frames = frames_uint8.shape[0]
            fps = 25
            compute_secs = max(int(num_frames / fps), 1)

            # ----- Step F: Call KeySync’s infer_raw.sh via our wrapper ----- #
            # We pass video_dir=run_dir (which contains input.mp4),
            # audio_dir=run_dir (which contains input.wav).
            infer.run_inference(
                video_dir=run_dir,
                audio_dir=run_dir,
                output_dir=output_frame_dir,
                keyframe_ckpt=keyframe_ckpt,
                interpolation_ckpt=interpolation_ckpt,
                compute_until=compute_secs,
                fix_occlusion=False,
                position=None,
                start_frame=0
            )

            # ----- Step G: Load back KeySync’s output PNGs into a Tensor ----- #
            png_files = sorted(
                f for f in os.listdir(output_frame_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            )
            if len(png_files) == 0:
                raise RuntimeError(f"KeySyncWrapper: No output frames found in {output_frame_dir}.")

            loaded_list = []
            for fname in png_files:
                img_path = os.path.join(output_frame_dir, fname)
                img = Image.open(img_path).convert("RGB")
                arr = np.array(img).astype("float32") / 255.0  # [H,W,3], float32
                tensor_frame = torch.from_numpy(arr)  # [H,W,3]
                loaded_list.append(tensor_frame)
                img.close()

            synced_tensor = torch.stack(loaded_list, dim=0)  # [N,H,W,3]

            # Return as a single IMAGE Tensor (ComfyUI expects that shape)
            return (synced_tensor,)

        finally:
            # Clean up everything under run_dir
            try:
                shutil.rmtree(run_dir, ignore_errors=True)
            except:
                pass

# Register this node for ComfyUI
NODE_CLASS_MAPPINGS = {
    "KeySyncWrapper": KeySyncWrapper
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KeySyncWrapper": KeySyncWrapper
}
