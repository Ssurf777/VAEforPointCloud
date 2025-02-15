import psutil

def get_available_memory():
    """Returns the available system memory in GB."""
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    return available_gb