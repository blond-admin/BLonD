import os
import shutil

# Exclude specific directories from deletion
exclude = (
    os.path.normpath("./.git"),
    os.path.normpath("./.venv"),
)


def is_effectively_empty(path):
    """Return True if directory is empty or contains only empty __pycache__ dirs."""
    try:
        entries = os.listdir(path)
        for entry in entries:
            full_path = os.path.join(path, entry)
            if os.path.isdir(full_path):
                if entry == "__pycache__":
                    # Check if __pycache__ is empty or only contains .pyc files
                    cache_contents = os.listdir(full_path)
                    if any(not f.endswith(".pyc") for f in cache_contents):
                        return False
                else:
                    return False
            else:
                return False
        return True  # Only __pycache__ or nothing
    except Exception:
        return False


def delete_empty_dirs(root_path):
    for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
        if (
            dirpath.startswith("./.git")
            or dirpath.startswith("./.venv")
            or dirpath.startswith("./.idea")
            or dirpath.startswith("./__doc")
        ):
            continue

        norm_dirpath = os.path.normpath(dirpath)
        if norm_dirpath in exclude:
            continue
        if not filenames and is_effectively_empty(dirpath):
            try:
                # Remove __pycache__ if present and empty
                pycache_path = os.path.join(dirpath, "__pycache__")
                if os.path.isdir(pycache_path):
                    try:
                        shutil.rmtree(pycache_path)
                        print(f"Deleted __pycache__: {pycache_path}")
                    except OSError:
                        pass  # Not empty, skip
                os.rmdir(dirpath)
                print(f"Deleted empty directory: {dirpath}")
            except Exception as e:
                print(f"Failed to delete {dirpath}: {e}")


# Example usage
root_directory = "./"
delete_empty_dirs(root_directory)
