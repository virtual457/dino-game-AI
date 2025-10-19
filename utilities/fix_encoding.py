# Run this before main.py to fix emoji encoding
import sys
if sys.platform == 'win32':
    import os
    os.system('chcp 65001 >nul')
