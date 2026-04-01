#!/usr/bin/env python3
"""
Simple launcher script for Smart Parking System
Run this with: python run.py
"""

import sys
import os
import tkinter as tk

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load the main module code and inject __name__ == "__main__"
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "import cv2.py"), encoding="utf-8") as f:
    source = f.read()

# Execute all the class/function definitions
exec(compile(source, "import cv2.py", "exec"), {"__name__": "__main__"})
