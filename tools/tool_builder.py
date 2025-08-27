import sys
import io

def run_python(code):
    """Execute Python code and return the output or error."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        exec(code, {})
        output = sys.stdout.getvalue()
        error = sys.stderr.getvalue()
        return output if not error else error
    except Exception as e:
        return str(e)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
