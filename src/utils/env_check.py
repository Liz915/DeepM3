import sys
import torch
import platform
import subprocess

def get_git_revision_short_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except:
        return "Not a git repo"

def print_env_fingerprint():
    """
    [Dev Note] 
    """
    print("="*40)
    print(" ENVIRONMENT FINGERPRINT")
    print("="*40)
    print(f"Time      : {subprocess.check_output(['date']).decode('utf-8').strip()}")
    print(f"OS        : {platform.system()} {platform.release()}")
    print(f"Python    : {sys.version.split()[0]}")
    print(f"PyTorch   : {torch.__version__}")
    
    # 
    if torch.cuda.is_available():
        print(f"Device    : CUDA ({torch.cuda.get_device_name(0)})")
        print(f"CUDNN     : {torch.backends.cudnn.version()}")
    elif torch.backends.mps.is_available():
        print(f"Device    : MPS (Apple Silicon)")
    else:
        print(f"Device    : CPU (Slow path warning)")

    print(f"Git Commit: {get_git_revision_short_hash()}")
    print("="*40)

if __name__ == "__main__":
    print_env_fingerprint()