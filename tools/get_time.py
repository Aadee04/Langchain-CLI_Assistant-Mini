import datetime

def get_time():
    """Return the current system time as a string."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
