import subprocess


def get_current_branch() -> str:
    return subprocess.check_output(
        ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
        stderr=subprocess.DEVNULL
    ).strip().decode()


def is_master_or_dev_branch() -> bool:
    return get_current_branch() in ("develop", "master")