from functools import wraps

# Modes available:
# 'disabled'
# 'tracing'
mode = 'disabled'
times = {}


def init(*args, **kw):
    pass


def finalize():
    pass


def traceit(*args, **kw):
    def decorator(f):
        @wraps(f)
        def timed(*args, **kw):
            return f(*args, **kw)

        return timed

    return decorator


class traced_region:

    def __init__(self, *args, **kw):
        pass

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        pass


def timeit(*args, **kw):
    def decorator(f):
        @wraps(f)
        def timed(*args, **kw):
            return f(*args, **kw)

        return timed

    return decorator


class timed_region:

    def __init__(self, *args, **kw):
        pass

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        pass


def start_timing(*args, **kw):
    pass


def stop_timing(*args, **kw):
    pass


def report(*args, **kw):
    pass


def reset(*args, **kw):
    pass
