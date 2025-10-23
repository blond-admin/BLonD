# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Sanity check. Run before committing**

:Authors: **Helga Timko, Konstantinos Iliakis**
'''

import argparse
import subprocess
import sys
import textwrap


def get_modified_files():
    """Returns a list containing all the modified or commited files tracked by git.

    Returns:
        list: list of file names
    """
    output = b''
    ret = subprocess.run(['git diff --no-commit-id --name-only -r HEAD'],
                         check=True, capture_output=True, shell=True)
    output += ret.stdout
    ret = subprocess.run(['git diff-tree --no-commit-id --name-only -r HEAD'],
                         check=True, capture_output=True, shell=True)
    output += ret.stdout
    files = output.decode().splitlines()
    files = [f for f in files if f.endswith('.py')]
    return files


class SanityCheck:
    """SanityCheck class
    """
    print_prompt = '==== SANITY_CHECK:'
    isort_settings = '--check --diff'
    flake8_settings = '--ignore E501,W503,W504,W291'
    pylint_settings = '-f colorized'
    pytest_settings = '-v'
    coverage_settings = '--cov'

    def __init__(self, all_tools=False, docs=False, pytest=None,
                 flake8=None, pylint=None, isort=None, coverage=False):
        """Constructor.

        Args:
            all_tools (bool, optional): _description_. Defaults to False.
            docs (bool, optional): _description_. Defaults to False.
            pytest (_type_, optional): _description_. Defaults to None.
            flake8 (_type_, optional): _description_. Defaults to None.
            pylint (_type_, optional): _description_. Defaults to None.
            isort (_type_, optional): _description_. Defaults to None.
            coverage (bool, optional): _description_. Defaults to False.
        """
        if all_tools:
            docs = True
            if flake8 is None:
                flake8 = './blond'
            if pytest is None:
                pytest = './unittests'
            if pylint is None:
                pylint = './blond'
            if isort is None:
                isort = './blond'

        if docs:
            self.compile_docs()
        if flake8:
            self.run_flake8(flake8)
        if pylint:
            self.run_pylint(pylint)
        if isort:
            self.run_isort(isort)
        if pytest:
            self.run_pytest(pytest, coverage)

        print(f"{self.print_prompt} Finished all checks!")

    def compile_docs(self):
        """Compile the documentation in html format.
        """
        print(f"{self.print_prompt} Compiling the documentation")

        ret = subprocess.run(['make -C __doc html'], shell=True, check=False,
                             capture_output=False)

        if ret.returncode != 0:
            print(f"{self.print_prompt} Documentation compilation failed")
        else:
            print(f"{self.print_prompt} Documentation compiled\n")

    def run_pylint(self, pylint):
        """Use pylint to report potential code syntax issues.

        Args:
            pylint (_type_): _description_
        """
        print(f"{self.print_prompt} Using pylint to report code syntax issues")

        files = ''

        if pylint == 'git':
            print(f"{self.print_prompt} Executing pylint on committed/modified files")
            files = ' '.join(get_modified_files())
        elif isinstance(pylint, str):
            files = pylint

        subprocess.run([f'{sys.executable} -m pylint {self.pylint_settings} {files}'],
                       shell=True, check=False, capture_output=False)

    def run_flake8(self, flake8):
        """Use flake8 to report code styling issues.

        Args:
            flake8 (_type_): _description_
        """
        print(f"{self.print_prompt} Using flake8 to report code style issues")

        files = ''

        if flake8 == 'git':
            print(f"{self.print_prompt} Executing flake8 on committed/modified files")
            files = ' '.join(get_modified_files())
        elif isinstance(flake8, str):
            files = flake8

        subprocess.run([f'{sys.executable} -m flake8 {self.flake8_settings} {files}'],
                       shell=True, check=False, capture_output=False)

    def run_isort(self, isort):
        """Use isort to sort imports.

        Args:
            isort (bool): _description_
        """
        print(f"{self.print_prompt} Using isort to sort the imports")

        files = ''

        if isort == 'git':
            print(f"{self.print_prompt} Executing isort on committed/modified files")
            files = ' '.join(get_modified_files())
        elif isinstance(isort, str):
            files = isort

        subprocess.run([f'{sys.executable} -m isort {self.isort_settings} {files}'],
                       shell=True, check=False, capture_output=False)

    def run_pytest(self, pytest, coverage):
        """Use pytest to run the unittests.

        Args:
            pytest (str): A space separated list of unittest names to run.
            coverage (bool): If True, report code coverage.
        """
        print(
            f"{self.print_prompt} Running the unittests to test the code's correctness")

        files = pytest

        if coverage:
            settings = f'{self.coverage_settings} {self.pytest_settings}'
        else:
            settings = self.pytest_settings

        subprocess.run([f'{sys.executable} -m pytest {settings} {files}'],
                       shell=True, check=False, capture_output=False)


def main():
    """Main function
    """
    # Arguments read from command line
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''
        SANITY CHECKER; run before committing from BLonD folder
        E.g. > python sanity_check.py -flake8 "llrf/signal_processing.py beam/profile.py"
             > python sanity_check.py -pytest ../unittests/general
             > python sanity_check.py --all
             > python sanity_check.py -pylint git (pylint report only committed or modified files)
             > python sanity_check.py -isort (use isort to sort the imports of the entirey BLonD library)
        '''))

    parser.add_argument('-all', '--all', dest='all', action='store_true',
                        help='Execute all checks', default=False)

    parser.add_argument('-docs', '--docs', dest='docs', action='store_true',
                        help='Compile docs in html format', default=False)

    parser.add_argument('-pytest', '--pytest', dest='pytest', const='./unittests',
                        nargs='?', type=str, default=None,
                        help='Run all unit-tests (default) or only ' +
                        'unit-tests found in the given files/directories')

    parser.add_argument('-coverage', '--coverage', dest='coverage',
                        action='store_true', default=False,
                        help='Report code coverage. Requires specifying the -pytest option.')

    parser.add_argument('-flake8', '--flake8', dest='flake8', const='./blond',
                        nargs='?', type=str, default=None,
                        help='Run the flake8 tool; on the entire BLonD library (default),' +
                        ' or on the specified files' +
                        ' or on committed/ modified files with -flake8 git')

    parser.add_argument('-pylint', '--pylint', dest='pylint', const='./blond',
                        nargs='?', type=str, default=None,
                        help='Run the pylint tool; on the entire BLonD library (default),' +
                        ' or on the specified files' +
                        ' or on committed/ modified files with -pylint git')

    parser.add_argument('-isort', '--isort', dest='isort', const='./blond',
                        nargs='?', type=str, default=None,
                        help='Run the isort tool; on the entire BLonD library (default),' +
                        ' or on the specified files' +
                        ' or on committed/ modified files with -isort git')

    args = parser.parse_args()

    # Passing a default option if no argument is given
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # Call the actual sanity check
    SanityCheck(args.all, args.docs, args.pytest, args.flake8,
                args.pylint, args.isort, args.coverage)


if __name__ == "__main__":
    main()
