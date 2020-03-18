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

import sys
import argparse
import os
import textwrap
import subprocess


class SanityCheck(object):

    def __init__(self, allChecks=None, docs=None,
                 pep8Files=None, unitTests=None):

        if allChecks:
            docs = True
            if pep8Files is None:
                pep8Files = ''
            if unitTests is None:
                unitTests = ''

        print("*** START SANITY CHECK ***")
        if unitTests is not None:
            self.unit_test(unitTests)
        if pep8Files is not None:
            self.pep8_test(pep8Files)
        if docs:
            self.compile_docs()
        print("*** END SANITY CHECK ***")

    def compile_docs(self):

        print("COMPILING DOCUMENTATION...")
        if os.path.isdir("__doc"):
            os.chdir("__doc")
        else:
            os.chdir("../__doc")
        os.system("make html")
        os.chdir("..")
        print("DOCUMENTATION COMPILED")
        print("")

    def pep8_test(self, pep8Files):

        # Ignore W291 trailing whitespace
        # Ignore W293 blank line contains whitespace
        # Ignore W391 blank line at end of file
        # Ignore E303 too many blank lines (3)
        # Ignore E128 continuation line under-indented for visual indent

        def command(x):
            try:
                print("~~~ EXECUTING PEP8 CHECK ON: %s ~~~" % x)
                subprocess.check_output(
                    ['pycodestyle', '--ignore', 'W291,W293,W391,E303,E128', x])
            except subprocess.CalledProcessError as e:
                print(e.output.decode())
        files = []
        if isinstance(pep8Files, str) and (pep8Files == ''):
            print("EXECUTING PEP8 CHECK ON THE ENTIRE BLOND PROJECT\n")

            for path, _, Files in os.walk("."):
                if ("./." in path) or ("./__" in path) or \
                        (".\." in path) or (".\__" in path):
                    continue
                for fileName in Files:
                    if fileName.endswith(".py") \
                            and not fileName.endswith("__.py"):
                        files.append(os.path.join(path, fileName))

        elif isinstance(pep8Files, str) and (pep8Files == 'git'):
            print("EXECUTING PEP8 CHECK ON THE COMMITTED/MODIFIED FILES\n")
            output = subprocess.check_output(['git', 'diff', '--no-commit-id',
                                              '--name-only', '-r', 'HEAD'])
            output += subprocess.check_output(['git', 'diff-tree',
                                               '--no-commit-id', '--name-only',
                                               '-r', 'HEAD'])
            files = output.decode().splitlines()
            files = [f for f in files if f.endswith('.py')]

        elif isinstance(pep8Files, str):
            files = pep8Files.split(' ')

        for file in files:
            command(file)

        print("PEP8 CHECK FINISHED\n")

    def unit_test(self, unitTests):

        def command(x):
            print("~~~ EXECUTING UNITTESTS FOUND IN %s ~~~" % x)
            try:
                subprocess.check_output(['python', x])
            except subprocess.CalledProcessError as e:
                print(e.output.decode())

        # Run unittests
        print("EXECUTING UNITTESTS")
        tests = []
        if isinstance(unitTests, str) and (unitTests == ''):
            for path, _, files in os.walk("../unittests"):
                tests += [os.path.join(path, file)
                          for file in files if file.startswith('test')]
        elif isinstance(unitTests, str):
            for test in unitTests.split(' '):
                if os.path.isdir(test):
                    for path, _, files in os.walk(test):
                        tests += [os.path.join(path, file)
                                  for file in files if file.startswith('test')]
                else:
                    tests.append(test)

        for test in tests:
            command(test)
        print("UNIT-TESTS FINISHED")
        print("")


def main():

    # Arguments read from command line
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''
        SANITY CHECKER; run before committing from BLonD folder
        E.g. > python sanity_check.py -p "llrf/signal_processing.py beam/profile.py"
             > python sanity_check.py -u "../unittests/general"
             > python sanity_check.py -a
             > python sanity_check.py -p "git" (pep8 report only committed or modified files)
        '''))
    parser.add_argument('-a', '--all', dest='all', action='store_true',
                        help='Execute all checks', default=False)
    parser.add_argument('-d', '--docs', dest='docs', action='store_true',
                        help='Compile docs in html format', default=False)
    parser.add_argument('-p', '--pep8', dest='pep8Files', const='',
                        nargs='?', type=str, default=None,
                        help='Run PEP8 check; on every BLonD file (default),' +
                        ' on the specified files' +
                        ' or on committed/ modified files with -p git')
    parser.add_argument('-u', '--unitTest', dest='unitTests',
                        const='', nargs='?', type=str, default=None,
                        help='Run all unit-tests (default) or only ' +
                        'unit-tests found in the given files/directories')

    args = parser.parse_args()

    # Passing a default option if no argument is given
    if len(sys.argv) == 1:
        if os.path.isdir('unittests'):
            args.unitTests = 'unittests'
        else:
            args.unitTests = '../unittests'

    # Call the actual sanity check
    SanityCheck(args.all, args.docs, args.pep8Files, args.unitTests)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
