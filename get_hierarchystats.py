#!/usr/bin/env python

import os
import sys
import json
import random
import argparse
import logging


logger = logging.getLogger(__name__)


LOG_FORMAT = "%(asctime)-15s %(levelname)s %(relativeCreated)dms " \
             "%(filename)s::%(funcName)s():%(lineno)d %(message)s"


def _parse_arguments(desc, args):
    """
    Parses command line arguments
    :param desc:
    :param args:
    :return:
    """
    help_fm = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=help_fm)
    parser.add_argument('input', help='Input CDRESULT file')
    parser.add_argument('--logconf', default=None,
                        help='Path to python logging configuration file in '
                             'this format: https://docs.python.org/3/library/'
                             'logging.config.html#logging-config-fileformat '
                             'Setting this overrides -v parameter which uses '
                             ' default logger. (default None)')
    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='Increases verbosity of logger to standard '
                             'error for log messages in this module '
                             '. Messages are '
                             'output at these python logging levels '
                             '-v = ERROR, -vv = WARNING, -vvv = INFO, '
                             '-vvvv = DEBUG, -vvvvv = NOTSET (default no '
                             'logging)')

    return parser.parse_args(args)


def _setup_logging(args):
    """
    Sets up logging based on parsed command line arguments.
    If args.logconf is set use that configuration otherwise look
    at args.verbose and set logging for this module and the one
    in ndexutil specified by TSV2NICECXMODULE constant
    :param args: parsed command line arguments from argparse
    :raises AttributeError: If args is None or args.logconf is None
    :return: None
    """

    if args.logconf is None:
        level = (50 - (10 * args.verbose))
        logging.basicConfig(format=LOG_FORMAT,
                            level=level)
        logger.setLevel(level)
        return

    # logconf was set use that file
    logging.config.fileConfig(args.logconf,
                              disable_existing_loggers=False)


class ClusterCounter(object):
    """

    """
    def __init__(self):
        """
        constructor
        """
        pass

    def get_cluster_count(self, input_stream):
        """

        :param netx_network:
        :return:
        """
        clusterset = set()
        nodes = input_stream.read().split(';')
        for anode in nodes:
            asplitnode = anode.split(',')
            if len(asplitnode) != 3:
                continue
            if asplitnode[2][0] == 'c':
                clusterset.add(asplitnode[0])
            if asplitnode[2][2] == 'c':
                clusterset.add(asplitnode[1])
        return len(clusterset)


def run(theargs, clustercounter=None):
    """
    Main flow of processing

    :param theargs:
    :return:
    """
    if clustercounter is None:
        clustercounter = ClusterCounter()

    with open(theargs.input, 'r') as f:
        count = clustercounter.get_cluster_count(f)
        sys.stdout.write(str(count) + '\n')
        sys.stdout.flush()
        return 0
    return 1


def main(args):
    """
    Main entry point for program
    :param args:
    :return:
    """
    desc = """
    
    Counts clusters in CDRESULT output

    """
    theargs = _parse_arguments(desc, args[1:])
    theargs.program = args[0]

    try:
        _setup_logging(theargs)
        return run(theargs)
    except Exception as e:
        logger.exception('Caught exception')
        return 2
    finally:
        logging.shutdown()


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(sys.argv))
