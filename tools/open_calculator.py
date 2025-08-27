import os
import subprocess

def open_calculator():
    """Open the calculator app (Windows only)."""
    if os.name == "nt":
        subprocess.Popen(["calc.exe"])
        return "Calculator opened."
    else:
        return "Calculator opening not supported on this OS."
