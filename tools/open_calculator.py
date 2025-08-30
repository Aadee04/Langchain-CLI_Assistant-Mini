from langchain.tools import tool
import os
import subprocess

@tool("open_calculator", return_direct=True)
def open_calculator(query: str = None) -> str:
    """Open the calculator app (Windows only). Input is empty. Output is always successful."""
    if os.name == "nt":
        subprocess.Popen(["calc.exe"])
        return "Calculator opened."
    else:
        return "Calculator opening not supported on this OS."
