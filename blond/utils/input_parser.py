import argparse
import sys

parser = argparse.ArgumentParser(description='BLonD simulation mainfile.')

# parser.add_argument('-w', '--workers', type=int, default=3,
#                     help='Number of worker processes to spawn.'
#                     '\nDefault: 3 (3 workers + 1 master)')

parser.add_argument('-p', '--particles', type=int,
                    help='Number of macro-particles.')

parser.add_argument('-s', '--slices', type=int,
                    help='Number of slices.')

parser.add_argument('-b', '--bunches', type=int,
                    help='Number of bunches.')

parser.add_argument('-reduce', '--reduce', type=int,
                    help='Number of turns to reduce.')


parser.add_argument('-t', '--turns', type=int,
                    help='Number of simulation turns.')

parser.add_argument('-mtw', '--mtw', type=int,
                    help='Number of simulation turns.')

parser.add_argument('-addload', '--addload', type=float, default=0.0,
                    help='Additional Load for tasks close to the master.')


parser.add_argument('-beginafter', '--beginafter', type=int,
                    help='Start the approximation after so many turns.')


parser.add_argument('-approx', '--approx', type=str, choices=['0', '1', '2'],
                    help='Which approximation to use: 0 (No approx), 1 (global reduce), 2 (scale histo).')


parser.add_argument('-o', '--omp', type=int, default=1,
                    help='Number of openmp threads to use.'
                    '\nDefault: 1')

parser.add_argument('-l', '--log', action='store_true',
                    help='Log debug messages or not.'
                    '\nDefault: Do not log.')

parser.add_argument('-logdir', '--logdir', type=str, default='./logs/',
                    help='Directory to store the log files.'
                    '\nDefault: ./logs.')


parser.add_argument('-time', '--time', action='store_true',
                    help='Time the specified regions of interest.'
                    '\nDefault: No timing.')

parser.add_argument('-timedir', '--timedir', type=str, default='./timings/',
                    help='Directory to store the timing reports.'
                    '\nDefault: ./timings')

parser.add_argument('-m', '--monitor', type=int, default=0,
                    help='Monitoring interval (0: no monitor).'
                    '\nDefault: 0')

parser.add_argument('-monitorfile', '--monitorfile', type=str, default=None,
                    help='h5 file to store the monitoring data.'
                    '\nDefault: Descriptive name based on simulation config.')


parser.add_argument('-seed', '--seed', type=int, default=None,
                    help='Seed value for the particle distribution generation.'
                    '\nDefault: None')


parser.add_argument('-trace', '--trace', action='store_true',
                    help='Trace the specified regions of interest (MPE).'
                    '\nDefault: No tracing.')

parser.add_argument('-tracefile', '--tracefile', type=str, default='mpe-trace',
                    help='The file name to save the MPE trace (without the file extension).'
                    '\nDefault: mpe-trace')


# parser.add_argument('-d', '--debug', action='store_true',
#                     help='Run workers in debug mode.'
#                     '\nDefault: No')


def parse():
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    args = parser.parse_args()
    print('Parsed arguments: ', args)
