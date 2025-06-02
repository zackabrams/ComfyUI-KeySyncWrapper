import os
import sys
import subprocess
import importlib

def install_requirements():
    """Install requirements for KeySync wrapper"""
    
    requirements = [
        "torch>=1.9.0",
        "torchvision>=0.10.0", 
        "torchaudio>=0.9.0",
        "opencv-python>=4.5.0",
        "numpy>=1.21.0",
        "pillow>=8.3.0",
        "librosa>=0.8.1",
        "scipy>=1.7.0",
        "resampy>=0.3.1",
        "numba>=0.56.0",
    ]
    
    for requirement in requirements:
        try:
            # Try to import first
            module_name = requirement.split(">=")[0].split("==")[0]
            if module_name == "opencv-python":
                module_name = "cv2"
            
            importlib.import_module(module_name)
            print(f"✓ {module_name} already installed")
            
        except ImportError:
            print(f"Installing {requirement}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", requirement
            ])

def clone_keysync():
    """Clone KeySync repository if not present"""
    
    keysync_dir = os.path.join(os.path.dirname(__file__), "keysync")
    
    if not os.path.exists(keysync_dir):
        print("Cloning KeySync repository...")
        try:
            subprocess.check_call([
                "git", "clone",
                "https://github.com/antonibigata/keysync.git", 
                keysync_dir
            ])
            print("✓ KeySync cloned successfully")
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone KeySync: {e}")
            return False
    else:
        print("✓ KeySync already present")
    
    return True

def setup_models_dir():
    """Setup models directory"""
    
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    print(f"✓ Models directory: {models_dir}")

if __name__ == "__main__":
    print("Setting up ComfyUI-KeySyncWrapper...")
    
    print("\n1. Installing requirements...")
    install_requirements()
    
    print("\n2. Cloning KeySync...")
    if not clone_keysync():
        print("WARNING: KeySync cloning failed. Manual setup may be required.")
    
    print("\n3. Setting up models directory...")
    setup_models_dir()
    
    print("\n✓ Setup complete!")
    print("\nTo use KeySync nodes:")
    print("1. Restart ComfyUI")
    print("2. Look for KeySync nodes in the node menu") 
    print("3. The first run will download models automatically")
