from datetime import datetime


def filesafe_datetime():
    # Get current datetime
    now = datetime.now()

    # Format it for filename use (e.g., 2025-06-24_145230)
    filename_safe_date = now.strftime("%Y-%m-%d_%H%M%S")
    return filename_safe_date