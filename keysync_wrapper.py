import os
import subprocess
import json
import logging
import math
from typing import Dict, Any, Optional

class KeySyncCLIWrapper:
    """
    Wrapper around KeySync's CLI scripts - uses actual upstream code with correct flags
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.keysync_dir = config['keysync_path']
        self.models_dir = config['models_path']
        self._validate_installation()
    
    def _validate_installation(self):
        """Verify KeySync is properly installed"""
        required_files = [
            'scripts/inference.sh',
            'model/keyframes.py',
            'model/interpolation.py'
        ]
        
        for file_path in required_files:
            full_path = os.path.join(self.keysync_dir, file_path)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"KeySync installation incomplete: {full_path}")
        
        # Check for model files
        model_files = ['keyframe_dub.pt', 'interpolation_dub.pt']
        for model_file in model_files:
            model_path = os.path.join(self.models_dir, model_file)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"KeySync model not found: {model_path}")
    
    def process_lipsync(self, video_path: str, audio_path: str, 
                       output_dir: str, face_enhance: bool = True) -> str:
        """
        Run KeySync lip synchronization using the actual CLI with correct flags
        """
        try:
            # Step 1: Merge audio back into video for KeySync's single-file approach
            merged_input = os.path.join(output_dir, "merged_input.mp4")
            self._merge_video_audio(video_path, audio_path, merged_input)
            
            # Step 2: Create filelist.txt as required by inference.sh
            filelist_path = self._create_filelist(merged_input, output_dir)
            
            # Step 3: Set up output subdirectory
            keysync_output_dir = os.path.join(output_dir, "keysync_output")
            os.makedirs(keysync_output_dir, exist_ok=True)
            
            # Step 4: Build correct CLI command matching actual KeySync usage
            cmd = self._build_inference_command(
                filelist_path, keysync_output_dir, face_enhance
            )
            
            # Step 5: Execute KeySync CLI
            result = subprocess.run(
                cmd,
                cwd=self.keysync_dir,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout for longer videos
            )
            
            if result.returncode != 0:
                logging.error(f"KeySync CLI stderr: {result.stderr}")
                logging.error(f"KeySync CLI stdout: {result.stdout}")
                raise RuntimeError(f"KeySync processing failed: {result.stderr}")
            
            # Step 6: Locate the final output file
            final_output = self._find_final_output(keysync_output_dir, merged_input)
            
            if not os.path.exists(final_output):
                raise FileNotFoundError(f"KeySync output not found: {final_output}")
            
            logging.info(f"KeySync processing completed: {final_output}")
            return final_output
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("KeySync processing timed out")
        except Exception as e:
            logging.error(f"KeySync wrapper error: {e}")
            raise e
    
    def _merge_video_audio(self, video_path: str, audio_path: str, output_path: str):
        """
        Merge separate video and audio files into single MP4 for KeySync
        """
        ffmpeg_cmd = [
            'ffmpeg', '-y',  # Overwrite output
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',  # Copy video stream as-is
            '-c:a', 'aac',   # Encode audio as AAC
            '-shortest',     # Match shortest stream duration
            output_path
        ]
        
        try:
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            logging.info(f"Successfully merged video+audio: {output_path}")
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg merge failed: {e.stderr}")
            raise RuntimeError(f"Failed to merge video and audio: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Please install FFmpeg.")
    
    def _create_filelist(self, merged_video_path: str, output_dir: str) -> str:
        """
        Create filelist.txt as required by KeySync's inference.sh
        """
        filelist_path = os.path.join(output_dir, "filelist.txt")
        
        try:
            with open(filelist_path, 'w') as f:
                f.write(f"{merged_video_path}\n")
            
            logging.info(f"Created filelist: {filelist_path}")
            return filelist_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to create filelist: {e}")
    
    def _build_inference_command(self, filelist_path: str, output_folder: str, 
                                face_enhance: bool) -> list:
        """
        Build the actual KeySync CLI command with correct flags
        """
        # Compute video duration for --compute_until flag
        duration = self._compute_video_duration(filelist_path)
        
        cmd = [
            'bash', 
            os.path.join(self.keysync_dir, 'scripts', 'inference.sh'),
            '--file_list', filelist_path,
            '--output_folder', output_folder,
            '--keyframes_ckpt', os.path.join(self.models_dir, 'keyframe_dub.pt'),
            '--interpolation_ckpt', os.path.join(self.models_dir, 'interpolation_dub.pt'),
            '--compute_until', str(duration)
        ]
        
        if face_enhance:
            cmd.extend(['--face_enhance', 'true'])
        
        logging.info(f"KeySync command: {' '.join(cmd)}")
        return cmd
    
    def _compute_video_duration(self, filelist_path: str) -> int:
        """
        Compute video duration from the first file in filelist for --compute_until
        """
        try:
            # Read first video from filelist
            with open(filelist_path, 'r') as f:
                video_path = f.readline().strip()
            
            if not video_path:
                return 10  # Default fallback
            
            # Use ffprobe to get duration
            ffprobe_cmd = [
                'ffprobe', 
                '-v', 'quiet', 
                '-print_format', 'json',
                '-show_format', 
                '-show_streams', 
                video_path
            ]
            
            result = subprocess.run(
                ffprobe_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            info = json.loads(result.stdout)
            
            # Try to get duration from format section first
            if 'format' in info and 'duration' in info['format']:
                duration_seconds = float(info['format']['duration'])
                return int(math.ceil(duration_seconds))
            
            # Fallback: check video stream duration
            for stream in info.get('streams', []):
                if stream.get('codec_type') == 'video' and 'duration' in stream:
                    duration_seconds = float(stream['duration'])
                    return int(math.ceil(duration_seconds))
            
            logging.warning("Could not determine video duration, using default")
            return 10
            
        except subprocess.CalledProcessError as e:
            logging.error(f"ffprobe failed: {e.stderr}")
            return 10
        except (json.JSONDecodeError, FileNotFoundError, Exception) as e:
            logging.error(f"Duration computation failed: {e}")
            return 10
    
    def _find_final_output(self, keysync_output_dir: str, merged_input_path: str) -> str:
        """
        Locate the final.mp4 output from KeySync's directory structure
        
        KeySync creates: output_folder/video_basename/final.mp4
        """
        # Get basename without extension for subdirectory name
        video_basename = os.path.splitext(os.path.basename(merged_input_path))[0]
        
        # KeySync creates a subdirectory named after the input video
        expected_subdir = os.path.join(keysync_output_dir, video_basename)
        expected_final = os.path.join(expected_subdir, "final.mp4")
        
        if os.path.exists(expected_final):
            return expected_final
        
        # Fallback: search for any final.mp4 in output directory
        for root, dirs, files in os.walk(keysync_output_dir):
            for file in files:
                if file == "final.mp4":
                    fallback_path = os.path.join(root, file)
                    logging.warning(f"Using fallback final.mp4 location: {fallback_path}")
                    return fallback_path
        
        # Last resort: search for any .mp4 file
        for root, dirs, files in os.walk(keysync_output_dir):
            for file in files:
                if file.endswith(".mp4"):
                    fallback_path = os.path.join(root, file)
                    logging.warning(f"Using fallback MP4 file: {fallback_path}")
                    return fallback_path
        
        raise FileNotFoundError(f"No output video found in {keysync_output_dir}")
