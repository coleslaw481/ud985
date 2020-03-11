#!/usr/bin/env python

import os
import sys
import argparse
import logging
import pandas
import matplotlib
import matplotlib.pyplot as plt


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
    parser.add_argument('csvfile', help='Input CSV file')
    parser.add_argument('--matplotlibgui', default='Qt4Agg',
                        help='Library to use for plotting')
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


def run(theargs):
    """
    Main flow of processing

    :param theargs:
    :return:
    """
    matplotlib.use(theargs.matplotlibgui)
    df = pandas.read_csv(theargs.csvfile, delimiter=',',
                         header=None)
    # sort by number of nodes then by number of edges
    df.sort_values(by=[2, 3], inplace=True)
    print(df.head())
    ax = df.plot(kind='scatter', x=2, y=1, color='red')
    ax.set_xlabel('# of Nodes')
    ax.set_ylabel('# of clusters')
    plt.show(block=False)

    # sort by number of edges then by number of nodes
    df.sort_values(by=[3, 2], inplace=True)
    print(df.head())
    ax = df.plot(kind='scatter', x=3, y=1, color='green')
    ax.set_xlabel('# of Edges')
    ax.set_ylabel('# of clusters')
    plt.title('hi')
    plt.show(block=False)

    # sort by density
    df.sort_values(by=[4,1], inplace=True)
    print(df.head())
    ax = df.plot(kind='scatter', x=4, y=1, color='blue')
    ax.set_xlabel('Density')
    ax.set_ylabel('# of clusters')
    plt.show(block=True)


    return 0


def main(args):
    """
    Main entry point for program
    :param args:
    :return:
    """
    desc = """
    
    Plots hierarchy stats

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
