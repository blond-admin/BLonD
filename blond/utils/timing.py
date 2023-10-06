from functools import wraps
import enum
try:
    from time import perf_counter as timer
except ImportError:
    from time import time as timer
import inspect
import numpy as np
import sys
import os
import pickle
import prettytable

class Mode(enum.Enum):
    OFF = 0
    ON = 1
    CUPY = 2
    LINEPROFILER = 3

mode = Mode(Mode.ON)

times = {}
start_time_stack = []
func_stack = []
excluded = ['lib_time']
lp = None


def timeit(key=None, exclude=False):
    global times, excluded, disabled, mode
    has_cupy = False
    if mode == Mode.CUPY:
        import cupy
        has_cupy = True
        start_gpu = cupy.cuda.Event()
        end_gpu = cupy.cuda.Event()
    
    if key is None:
        key = f'func_{len(times)}'

    def decorator(f):
        @wraps(f)
        def timed(*args, **kw):
            if mode == Mode.OFF:
                return f(*args, **kw)
            elif mode in [Mode.ON, Mode.CUPY]:
                ts = timer()
                if has_cupy:
                    start_gpu.record()
                result = f(*args, **kw)
                if has_cupy:
                    end_gpu.record()
                    end_gpu.synchronize()
                te = timer()

                if(key not in times):
                    times[key] = []
                    if exclude:
                        excluded.append(key)
                if has_cupy:
                    elapsed_time = cupy.cuda.get_elapsed_time(start_gpu, end_gpu)
                else:
                    elapsed_time = (te-ts) * 1000
                times[key].append(elapsed_time)
                if 'lib_time' not in times:
                    times['lib_time'] = []
                times['lib_time'].append((timer() - te) * 1000)

                return result
            elif mode == Mode.LINEPROFILER:
                from line_profiler import LineProfiler
                global lp
                if not lp:
                    lp = LineProfiler()
                lp_wrapper = lp(f)
                result = lp_wrapper(*args, **kw)
                return result
            else:
                raise RuntimeError(
                    '[timing.py:timeit] mode: %s not available' % mode)
        return timed
    return decorator



class timed_region:

    def __init__(self, key=None, is_child_function=False):
        global mode
        if mode == Mode.OFF:
            return
        elif mode in [Mode.ON, Mode.CUPY]:
            ls = timer()
            if mode == Mode.CUPY:
                import cupy
                self.start_gpu = cupy.cuda.Event()
                self.end_gpu = cupy.cuda.Event()
                self.cupy = True
            else:
                self.cupy = False
            global times, excluded
            if key:
                self.key = key
            else:
                parent = inspect.stack()[1]
                self.key = parent.filename.split('/')[-1]
                self.key = self.key + ':' + parent.lineno

            if (self.key not in times):
                times[self.key] = []

            if is_child_function == True:
                excluded.append(self.key)

            if 'lib_time' not in times:
                times['lib_time'] = []
            # times['lib_time'].append((timer() - ls) * 1000)
        else:
            raise RuntimeError(
                '[timing:timed_region] mode: %s not available' % mode)

    def __enter__(self):
        global mode

        if mode == Mode.OFF:
            return self
        elif mode in [Mode.ON, Mode.CUPY]:
            ls = timer()
            global times, excluded

            # times['lib_time'].append((timer() - ls) * 1000)

            self.ts = timer()
            if self.cupy:
                self.start_gpu.record()

            return self
        else:
            raise RuntimeError(
                '[timing:timed_region] mode: %s not available' % mode)

    def __exit__(self, type, value, traceback):
        global mode

        if mode == Mode.OFF:
            return
        elif mode in [Mode.ON, Mode.CUPY]:
            te = timer()
            global times, excluded
            if self.cupy:
                import cupy
                self.end_gpu.record()
                self.end_gpu.synchronize()
                elapsed_time = cupy.cuda.get_elapsed_time(self.start_gpu, self.end_gpu)
            else:
                elapsed_time = (te-self.ts)*1000
            times[self.key].append(elapsed_time)

            # times['lib_time'].append((timer() - te) * 1000)
            return
        else:
            raise RuntimeError(
                '[timing:timed_region] mode: %s not available' % mode)


def start_timing(funcname=''):
    global func_stack, start_time_stack, disabled, times, mode
    if mode == Mode.OFF:
        return
    elif mode in [Mode.ON, Mode.CUPY]:
        ts = timer()
        if funcname:
            key = funcname
        else:
            parent = inspect.stack()[1]
            key = parent.filename.split('/')[-1]
            key = key + ':' + parent.lineno
        func_stack.append(key)
        start_time_stack.append(timer())
        if 'lib_time' not in times:
            times['lib_time'] = []
        times['lib_time'].append((timer() - ts) * 1000)
    else:
        raise RuntimeError(
            '[timing:timed_region] mode: %s not available' % mode)


def stop_timing(exclude=False):
    global times, start_time_stack, func_stack, excluded, mode
    if mode == Mode.OFF:
        return
    elif mode in [Mode.ON, Mode.CUPY]:
        ts = timer()

        elapsed = (timer() - start_time_stack.pop()) * 1000
        key = func_stack.pop()
        if(key not in times):
            times[key] = []
            if exclude == True:
                excluded.append(key)
        times[key].append(elapsed)
        if 'lib_time' not in times:
            times['lib_time'] = []
        times['lib_time'].append((timer() - ts) * 1000)
    else:
        raise RuntimeError(
            '[timing:timed_region] mode: %s not available' % mode)


def report(skip=0, total_time=None, out_file=None, out_dir='./',
           save_pickle=False):
    global times, excluded, mode

    if mode == Mode.OFF:
        return
    elif mode in [Mode.ON, Mode.CUPY]:
        if out_file:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            out = open(os.path.join(out_dir, out_file), 'w')
        else:
            out = sys.stdout
            table = prettytable.PrettyTable()
            field_names = ['function', 'total time (sec)', 'time per call (ms)', 'std (%)', 'calls', 'global (%)']
            table.field_names = field_names
            table.float_format = '.3'
            # formats = {
            #     field_names[1]: lambda f, v: f"{v:.3f}",
            #     field_names[2]: lambda f, v: f"{v:.3f}",
            #     field_names[3]: lambda f, v: f"{v:.2f}",
            #     field_names[5]: lambda f, v: f"{v:.2f}"
            # }
            # table.custom_format = formats
            #formats = {field: lambda f, v: f"{v:{f}}" for field, fmt in zip(field_names, format_strings)}

            table.align = "l"

        if isinstance(total_time, str):
            _total_time = sum(times[total_time][skip:])
            excluded.append(total_time)
            _total_time -= sum(times['lib_time'])
        elif isinstance(total_time, float):
            _total_time = total_time
            _total_time -= sum(times['lib_time'])
        else:
            _total_time = sum(sum(x[skip:])
                              for k, x in times.items() if k not in excluded)

        otherPercent = 100.0
        otherTime = _total_time

        if out != sys.stdout:
            out.write(
                'function\ttotal_time(sec)\ttime_per_call(ms)\tstd(%)\tcalls\tglobal(%)\n')
            

        for k, v in sorted(times.items()):

            if k == 'lib_time':
                continue

            vSum = np.sum(v[skip:])
            vMean = vSum / len(v[skip:])
            vStd = np.std(v[skip:])
            vGPercent = 100 * vSum / _total_time

            if k not in excluded:
                otherPercent -= vGPercent
                otherTime -= vSum

            if out != sys.stdout:
                out.write('%s\t%.3lf\t%.3lf\t%.2lf\t%d\t%.2lf\n'
                        % (k, vSum/1000., vMean, 100.0 * vStd / vMean,
                            len(v), vGPercent))
            else:
                table.add_row([k, vSum/1000., vMean, 100.0 * vStd / vMean,
                            len(v), vGPercent])

        if out != sys.stdout:
            out.write('%s\t%.3lf\t%.3lf\t%.2lf\t%d\t%.2lf\n'
                    % ('other', otherTime/1000., otherTime, 0.0, 1, otherPercent))

            out.write('%s\t%.3lf\t%.3lf\t%.2lf\t%d\t%.2lf\n'
                    % ('total_time', (_total_time/1e3), _total_time, 0.0, 1, 100))
        else:
            table.add_row(['other', otherTime/1000., otherTime, 0.0, 1, otherPercent])
            table.add_row(['total_time', (_total_time/1e3), _total_time, 0.0, 1, 100])
            out.write(table.get_string() + "\n")
        
        if save_pickle and out_file:
            times['total_time'] = _total_time
            times['other'] = otherTime
            out_file = os.path.splitext(out_file)[0] + '.p'
            with open(os.path.join(out_dir, out_file), 'wb') as picklefile:
                pickle.dump(times, picklefile)


        if out_file:
            out.close()
    elif mode == Mode.LINEPROFILER:
        lp.print_stats()
    else:
        raise RuntimeError('[timing:report] mode: %s not available' % mode)


def reset():
    global times, start_time_stack, func_stack, excluded, mode, lp
    times = {}
    start_time_stack = []
    func_stack = []
    excluded = ['lib_time']
    # mode = 'timing'
    lp = None


def get(lst, exclude_lst=[]):
    global times, mode, excluded
    total = 0
    if mode != Mode.OFF:
        for k, v in times.items():
            if (k in excluded) or (k in exclude_lst):
                continue
            if np.any([l in k for l in lst]):
                total += np.sum(v)
    return total
