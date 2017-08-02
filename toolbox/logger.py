# coding: utf8
# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
**Logging messages**

:Authors: **Helga Timko**
'''

import logging, sys


class Logger(object):
    """Class to log messages coming from other classes. Messages contain 
    {Time stamp} {Class name} {Message}
    
    Parameters
    ----------
    info : bool
        Log INFO messages in stdout stream; default is True
    errors : bool
        Log WARNING and ERROR messages in stderr stream; default is True
    debug : bool
        Log DEBUG messages in stdout stream; default is False
    
    """
    def __init__(self, info = True, errors = True, debug = True):

        # Root logger
        root_logger = logging.getLogger('')
        #self.root_logger.setLevel(logging.INFO)

        # Define logging format
        log_format = logging.Formatter("{asctime} {name:<15s} {message}")
        
        # Output on console on INFO level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(log_format)
        # Add to root logger
        root_logger.addHandler(console_handler)
        logging.info("Logging initialized")
    

#         if info == True:
#             logging.basicConfig(format=self.format, datefmt='%m-%d %H:%M',
#                                 level=logging.INFO, stream=sys.stdout)
#         if errors == True:
#             logging.basicConfig(format=self.format, level=logging.ERROR, 
#                                 stream=sys.stderr)
#             logging.basicConfig(format=self.format, level=logging.WARNING, 
#                                 stream=sys.stderr)
        if debug == True:
            logging.basicConfig(level=logging.DEBUG, filename='debug.log',
                                format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
            logging.debug("Logging in DEBUG mode")
            
#         self.logger = logging.getLogger("BLonD")
#         self.logger.info("Logging initialized")
# 
# 
# # set up logging to file - see previous section for more details
# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
#                     datefmt='%m-%d %H:%M',
#                     filename='/temp/myapp.log',
#                     filemode='w')
# # define a Handler which writes INFO messages or higher to the sys.stderr
# # set a format which is simpler for console use
# formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# # tell the handler to use this format
# # add the handler to the root logger
# 
# # Now, we can log to the root logger, or any other logger. First the root...
# logging.info('Jackdaws love my big sphinx of quartz.')
