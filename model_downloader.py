import os
import subprocess
import json
from huggingface_hub import hf_hub_download, snapshot_download
import folder_paths
import logging

class KeySyncModelDownloader:
    """
    Downloads and sets up the actual KeySync repository and models
    """
    
    def __init__(self):
        self.base_dir = os.path.join(folder_paths.models_dir, "keysync")
        self.keysync_repo_dir = os.path.join(self.base_dir, "keysync_repo")
        self.models_dir = os.path.join(self.base_dir, "models")
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories"""
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
    
    def setup_keysync(self, variant: str = "base", force_reinstall: bool = False):
        """
        Download and setup KeySync repository and models
        """
        try:
            # Step 1: Clone/update KeySync repository
            self._setup_keysync_repo(force_reinstall)
            
            # Step 2: Download model checkpoints
            self._download_models(variant, force_reinstall)
            
            # Step 3: Install Python dependencies
            self._install_dependencies()
            
            # Step 4: Create configuration
            config = {
                'keysync_path': self.keysync_repo_dir,
                'models_path': self.models_dir,
                'variant': variant,
                'version': self._get_repo_version()
            }
            
            # Step 5: Validate installation
            self._validate_setup(config)
            
            logging.info(f"KeySync setup complete: {config}")
            return config
            
        except Exception as e:
            logging.error(f"KeySync setup failed: {e}")
            raise e
    
    def _setup_keysync_repo(self, force_reinstall: bool):
        """Clone or update KeySync repository"""
        if force_reinstall and os.path.exists(self.keysync_repo_dir):
            import shutil
            shutil.rmtree(self.keysync_repo_dir)
        
        if not os.path.exists(self.keysync_repo_dir):
            logging.info("Cloning KeySync repository...")
            subprocess.run([
                'git', 'clone', 
                'https://github.com/antonibigata/keysync.git',
                self.keysync_repo_dir
            ], check=True)
        else:
            logging.info("Updating KeySync repository...")
            subprocess.run([
                'git', 'pull'
            ], cwd=self.keysync_repo_dir, check=True)
    
    def _download_models(self, variant: str, force_reinstall: bool):
        """Download actual KeySync model files from HuggingFace"""
        model_files = [
            'keyframe_dub.pt',
            'interpolation_dub.pt'
        ]
        
        for model_file in model_files:
            local_path = os.path.join(self.models_dir, model_file)
            
            if force_reinstall or not os.path.exists(local_path):
                logging.info(f"Downloading {model_file}...")
                hf_hub_download(
                    repo_id="antonibigata/keysync",
                    filename=model_file,
                    local_dir=self.models_dir,
                    local_dir_use_symlinks=False
                )
        
        # Download any additional variant-specific files if they exist
        if variant == "hq":
            # Check if HQ variants exist on HuggingFace
            try:
                hq_files = ['keyframe_hq.pt', 'interpolation_hq.pt']
                for hq_file in hq_files:
                    local_path = os.path.join(self.models_dir, hq_file)
                    if force_reinstall or not os.path.exists(local_path):
                        hf_hub_download(
                            repo_id="antonibigata/keysync",
                            filename=hq_file,
                            local_dir=self.models_dir,
                            local_dir_use_symlinks=False
                        )
            except Exception:
                logging.warning("HQ models not available, using base models")
    
    def _install_dependencies(self):
        """Install KeySync Python dependencies"""
        requirements_path = os.path.join(self.keysync_repo_dir, 'requirements.txt')
        
        if os.path.exists(requirements_path):
            logging.info("Installing KeySync dependencies...")
            subprocess.run([
                'pip', 'install', '-r', requirements_path
            ], check=True)
        else:
            # Install known KeySync dependencies
            known_deps = [
                'torch>=1.9.0',
                'torchvision',
                'numpy',
                'opencv-python',
                'librosa',
                'soundfile',
                'transformers',
                'accelerate'
            ]
            
            for dep in known_deps:
                try:
                    subprocess.run(['pip', 'install', dep], check=True)
                except subprocess.CalledProcessError:
                    logging.warning(f"Failed to install {dep}")
    
    def _get_repo_version(self) -> str:
        """Get current KeySync repository version/commit"""
        try:
            result = subprocess.run([
                'git', 'rev-parse', 'HEAD'
            ], cwd=self.keysync_repo_dir, capture_output=True, text=True)
            
            return result.stdout.strip()[:8]  # Short commit hash
        except:
            return "unknown"
    
    def _validate_setup(self, config):
        """Validate that KeySync is properly set up"""
        from .preprocessing import check_ffmpeg_availability, check_ffprobe_availability
        
        # Check repository files
        required_files = [
            'model/keyframes.py',
            'model/interpolation.py',
            'scripts/inference.sh'
        ]
        
        for file_path in required_files:
            full_path = os.path.join(config['keysync_path'], file_path)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Missing KeySync file: {file_path}")
        
        # Check model files
        model_files = ['keyframe_dub.pt', 'interpolation_dub.pt']
        for model_file in model_files:
            model_path = os.path.join(config['models_path'], model_file)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Missing model file: {model_file}")
        
        # Check system dependencies
        if not check_ffmpeg_availability():
            raise RuntimeError("FFmpeg not found. Please install FFmpeg and ensure it's in your PATH.")
        
        if not check_ffprobe_availability():
            raise RuntimeError("FFprobe not found. Please install FFmpeg (includes FFprobe).")
        
        logging.info("KeySync validation passed")
