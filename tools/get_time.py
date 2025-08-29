from langchain.tools import tool
import datetime

@tool
def get_time(query: str) -> str:
    """Return the current system time as a string."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
