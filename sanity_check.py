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
import os
import textwrap
import subprocess


class SanityCheck(object):

    def __init__(self, allChecks=None, docs=None,
                 pep8Files=None, unitTests=None):

        if allChecks:
            docs = True
            if pep8Files is None:
                pep8Files = True
            if unitTests is None:
                unitTests = True

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
        os.chdir("__doc")
        os.system("make html")
        os.chdir("..")
        print("Documentation compiled")
        print("")

    def pep8_test(self, pep8Files):

        # Ignore W291 trailing whitespace
        # Ignore W293 blank line contains whitespace
        # Ignore W391 blank line at end of file
        def command(x):
            try:
                subprocess.check_output(
                    ['pep8', '--ignore', 'W291,W293,W391', x])
            except subprocess.CalledProcessError as e:
                print(e.output.decode())

        if pep8Files:
            for file in pep8Files.split(' '):
                print("EXECUTING PEP8 CHECK ON %s" % file)
                command(file)
        else:
            print("EXECUTING PEP8 CHECK ON ENTIRE BLOND DISTRIBUTION")
            for path, subDir, files in os.walk("."):
                if ("./." not in path) and ("./__" not in path) and \
                        (".\." not in path) and (".\__" not in path):
                    for fileName in files:
                        if fileName.endswith(".py") \
                                and not fileName.endswith("__.py"):  # \
                            pep8File = os.path.join(path, fileName)
                            command(pep8File)
        print("PEP8 check finished")
        print("")

    def unit_test(self, unitTests):

        def command(x):
            print("~~~ EXECUTING UNITTESTS FOUND IN %s ~~~" % x)
            try:
                subprocess.check_output(['python', x])
            except subprocess.CalledProcessError as e:
                print(e.output.decode())

        # Run unittests
        print("EXECUTING UNITTESTS...")
        tests = []
        if unitTests:
            for test in unitTests.split(' '):
                if os.path.isdir(test):
                    for path, subDir, files in os.walk(test):
                        tests += [os.path.join(path, file)
                                  for file in files if file.startswith('test')]
                else:
                    tests.append(test)
        else:
            for path, subDir, files in os.walk("unittests"):
                tests += [os.path.join(path, file)
                          for file in files if file.startswith('test')]
        for test in tests:
            command(test)
        print("Unit-tests finished")
        print("")


def main():

    # Arguments read from command line
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''
        SANITY CHECKER; run before committing from BLonD folder
        E.g. > python sanity_check.py -p 'llrf/signal_processing.py beam/profile.py'
        '''))
    parser.add_argument('-a', '--all', dest='all', action='store_true',
                        help='Execute all checks', default=False)
    parser.add_argument('-d', '--docs', dest='docs', action='store_true',
                        help='Compile docs in html format', default=False)
    parser.add_argument('-p', '--pep8', dest='pep8Files', const='',
                        nargs='?', type=str, default=None,
                        help='Run PEP8 check; on all files (default)' +
                        ' or on the specified files')
    parser.add_argument('-u', '--unitTest', dest='unitTests',
                        const='', nargs='?', type=str, default=None,
                        help='Run all unit-tests (default) or only ' +
                        'unit-tests found in the given files/directories')

    args = parser.parse_args()

    # Call the actual sanity check
    SanityCheck(args.all, args.docs, args.pep8Files, args.unitTests)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
