"""
Installation script for ComfyUI MickMumpitz Nodes
This script is called by ComfyUI Manager during installation
"""

import os
import subprocess
import sys


def install():
    """
    Install any additional dependencies required by the nodes.
    """
    print("Installing ComfyUI MickMumpitz Nodes dependencies...")

    # Install opencv-python for advanced color correction features
    try:
        import cv2
        print("opencv-python already installed")
    except ImportError:
        print("Installing opencv-python...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python>=4.5.0"])

    print("ComfyUI MickMumpitz Nodes installed successfully!")
    return True


if __name__ == "__main__":
    install()
