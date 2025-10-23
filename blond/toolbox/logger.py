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

import logging


class Logger:
    """Class to log messages coming from other classes. Messages contain 
    {Time stamp} {Class name} {Log level} {Message}. Errors, warnings and info
    are logged into the console. To disable logging, call Logger().disable()

    Parameters
    ----------
    debug : bool
        Log DEBUG messages in 'debug.log'; default is False

    """

    def __init__(self, debug=False):

        # Root logger on DEBUG level
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # Console handler on INFO level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        log_format = logging.Formatter(
            "%(asctime)s %(name)-25s %(levelname)-9s %(message)s")
        console_handler.setFormatter(log_format)
        root_logger.addHandler(console_handler)

        # File logger on DEBUG level
        if debug:
            file_handler = logging.FileHandler('debug.log', mode='w')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(log_format)
            root_logger.addHandler(file_handler)

        logging.info("Start logging")
        if debug:
            logging.debug("Logger in debug mode")

    def disable(self):
        """Disables all logging."""

        logging.info("Disable logging")
        logging.disable(level=logging.NOTSET)
