from datetime import datetime
import os
import inspect
import numpy as np


def filesafe_datetime() -> str:
    # Get current datetime
    now = datetime.now()

    # Format it for filename use (e.g., 2025-06-24_145230)
    filename_safe_date = now.strftime("%Y-%m-%d_%H%M%S")
    return filename_safe_date


def callers_relative_path(filename: str, stacklevel: int) -> str:
    # Get the path of the file that called this function
    caller_frame = inspect.stack()[stacklevel]
    caller_file = caller_frame.filename
    caller_dir = os.path.dirname(os.path.abspath(caller_file))

    # Build the full save path
    full_path = os.path.join(caller_dir, filename)
    return full_path
