import cProfile
import functools
import pstats
import os


def do_cprofile(output):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            DO_PROF = os.environ.get("PROFILING", "0") == "1"
            if DO_PROF:
                profile = cProfile.Profile()
                profile.enable()
                result = func(*args, **kwargs)
                profile.disable()
                ps = pstats.Stats(profile).sort_stats("tottime")
                ps.dump_stats(output)
            else:
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator
