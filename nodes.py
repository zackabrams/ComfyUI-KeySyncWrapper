import os
from videohelpersuite import VideoHelperSuite
import infer

class KeySyncWrapper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE_LIST",),
                "audio_wav": ("AUDIO_WAV",),
                "keypoint_confidence_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "sync_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01
                })
            }
        }

    RETURN_TYPES = ("IMAGE_LIST",)
    FUNCTION = "process"
    CATEGORY = "KeySync"

    def process(self, frames, audio_wav, keypoint_confidence_threshold, sync_strength):
        """
        1. Save incoming frames + audio to temporary folders.
        2. Call KeySync’s existing infer_raw.sh via infer.run_inference().
        3. Load back the synced frames and return them.
        """
        # 1) Create a temp directory (VideoHelperSuite manages cleanup)
        temp_root = VideoHelperSuite.get_temp_dir()
        input_frame_dir = os.path.join(temp_root, "input_frames")
        os.makedirs(input_frame_dir, exist_ok=True)
        VideoHelperSuite.save_frames(frames, input_frame_dir, fps=25)

        input_audio_path = os.path.join(temp_root, "input.wav")
        VideoHelperSuite.save_audio(audio_wav, input_audio_path, sample_rate=16000)

        # 2) Prepare output folder
        output_frame_dir = os.path.join(temp_root, "synced_frames")
        os.makedirs(output_frame_dir, exist_ok=True)

        # 3) Model checkpoints: assume user cloned KeySync’s pretrained models into
        #    <this_folder>/pretrained_models/
        model_dir = os.path.join(os.path.dirname(__file__), "pretrained_models")
        keyframe_ckpt = os.path.join(model_dir, "keyframe_dub.pt")
        interpolation_ckpt = os.path.join(model_dir, "interpolation_dub.pt")

        # 4) Compute how many seconds of video we have (min 1s to avoid zero)
        num_frames = len(frames)
        fps = 25
        compute_secs = max(int(num_frames / fps), 1)

        # 5) Call our infer.py wrapper
        infer.run_inference(
            video_dir=input_frame_dir,
            audio_dir=temp_root,
            output_dir=output_frame_dir,
            keyframe_ckpt=keyframe_ckpt,
            interpolation_ckpt=interpolation_ckpt,
            compute_until=compute_secs,
            fix_occlusion=False,
            position=None,
            start_frame=0
        )

        # 6) Load the synced frames back into a list of PIL images
        synced_frames = VideoHelperSuite.load_frames(output_frame_dir)
        return (synced_frames,)
