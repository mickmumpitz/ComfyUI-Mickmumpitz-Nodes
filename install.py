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
    Basic dependencies (torch, numpy, Pillow) are usually already available in ComfyUI.
    """
    print("Installing ComfyUI MickMumpitz Nodes...")

    # All required packages (torch, numpy, Pillow) are typically already installed with ComfyUI
    # If you need additional packages in the future, install them here:
    # subprocess.check_call([sys.executable, "-m", "pip", "install", "package-name"])

    print("ComfyUI MickMumpitz Nodes installed successfully!")
    return True


if __name__ == "__main__":
    install()
