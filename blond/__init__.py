import os

if ('posix' in os.name):
    libblond = ctypes.CDLL('cpp_routines/libblond.so')
elif ('win' in sys.platform):
    libblond = ctypes.CDLL('cpp_routines\\libblond.dll')
else:
    print('YOU DO NOT HAVE A WINDOWS OR LINUX OPERATING SYSTEM. ABORTING...')
    sys.exit()