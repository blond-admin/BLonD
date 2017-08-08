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

:Authors: **Helga Timko**
'''

import argparse
import os
import textwrap



class SanityCheck(object):
    
    def __init__(self, allChecks = None, docs = None, pep8Test = None, 
                 pep8File = None, unitTest = None):
        
        if allChecks:
            docs = True
            pep8Test = True
            unitTest = True
            
        print("*** START SANITY CHECK ***")
        if unitTest:
            self.unit_test()      
        if pep8Test:
            self.pep8_test(pep8File)            
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
        
        
    def pep8_test(self, pep8File):
        
        # Ignore W291 trailing whitespace
        # Ignore W293 blank line contains whitespace
        # Ignore W391 blank line at end of file
        command = lambda x: os.system("pep8 --ignore=W291,W293,W391 " + x)        

        if pep8File:
            try:
                print("EXECUTING PEP8 CHECK ON %s" %pep8File)
                command(pep8File)
            except:
                print("File to be checked for PEP8 not found")
        else:
            print("EXECUTING PEP8 CHECK ON ENTIRE BLOND DISTRIBUTION")
            for path, subDir, files in os.walk("."):
                if ("./." not in path) and ("./__" not in path) and \
                    (".\." not in path) and (".\__" not in path):
                    for fileName in files:
                        if fileName.endswith(".py") \
                            and not fileName.endswith("__.py"): # \
                                pep8File = os.path.join(path, fileName)
                                print("~~~ CHECK %s ~~~" %pep8File)
                                command(pep8File)
        print("PEP8 check finished")
        print("")


    def unit_test(self):
        
        # Run unittests
        print("EXECUTING UNITTESTS...")
        for folderName in os.listdir("unittests"):
            for fileName in os.listdir("unittests/" + folderName):
                if (fileName.startswith("test")):
                    
                    print("~~~ RUN %s ~~~" %fileName)
                    os.system("python unittests/" + folderName + "/" + fileName)
        print("Unit-tests finished")
        print("")
                    
    
    
def main():
    
    # Arguments read from command line
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''
        SANITY CHECKER; run before committing from BLonD folder
        E.g. > python sanity_check.py -p --pep8File llrf/signal_processing.py
        '''))    
    parser.add_argument('-a', dest ='allChecks ', action = 'store_true', 
                        help = 'Execute all checks')
    parser.set_defaults(allChecks = False)
    parser.add_argument('-d', dest = 'docs', action = 'store_true', 
                        help = 'Compile docs in html format')
    parser.set_defaults(docs = False)
    parser.add_argument('-p', dest = 'pep8Test', action = 'store_true', 
                        help='Run PEP8 check; on all files (default)'+
                        ' or on --pep8File only')
    parser.add_argument('--pep8File', type = str, 
                        help = 'File to run PEP8 check on', default = None)
    parser.set_defaults(pep8Test = False)
    parser.add_argument('-u', dest = 'unitTest', action = 'store_true', 
                        help = 'Run all unit-tests')
    parser.set_defaults(unitTest = False) 
               
    args = parser.parse_args()
    
    # Call the actual sanity check
    SanityCheck(args.allChecks, args.docs, args.pep8Test, args.pep8File, 
                args.unitTest)
           
        
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)



