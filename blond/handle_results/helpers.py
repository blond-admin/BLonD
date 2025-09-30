import inspect
import os
from datetime import datetime


def filesafe_datetime() -> str:
    """
    Datetime string suitable as filename
    """
    # Get current datetime
    now = datetime.now()

    # Format it for filename use (e.g., 2025-06-24_145230)
    filename_safe_date = now.strftime("%Y-%m-%d_%H%M%S")
    return filename_safe_date


def callers_relative_path(filename: str, stacklevel: int) -> str:
    """
    Absolute path according to filepath of the python script at given stacklevel

    Parameters
    ----------
    filename
        Local filepath, e.g. resources/file1.txt
    stacklevel
        Use global filepath according to the file at the level of
        the python call stack

    Returns
    -------
    Absolute path according to filepath of the python script at given stacklevel

    """
    # Get the path of the file that called this function
    caller_frame = inspect.stack()[stacklevel]
    caller_file = caller_frame.filename
    caller_dir = os.path.dirname(os.path.abspath(caller_file))

    # Build the full save path
    full_path = os.path.join(caller_dir, filename)
    return full_path
