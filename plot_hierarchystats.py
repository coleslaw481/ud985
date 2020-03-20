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


class Formatter(argparse.ArgumentDefaultsHelpFormatter,
                argparse.RawDescriptionHelpFormatter):
    pass


def _parse_arguments(desc, args):
    """
    Parses command line arguments
    :param desc:
    :param args:
    :return:
    """
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=Formatter)
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
    Using pandas this function reads in the csvfile input file and
    generates 2 figures with subplots in each figure. The first
    figure contains scatter subplots and the second figure is a
    histogram plot for each column of the data.

    NOTE: In this implementation matplotlib will pause after
    creating the figures
    so the caller must close the figures for the program to exit

    :param theargs: arguments from ArgParse
    :return: 0 upon success otherwise failure
    """
    matplotlib.use(theargs.matplotlibgui)
    df = pandas.read_csv(theargs.csvfile, delimiter=',',
                         header=None)
    csvfilename = os.path.basename(theargs.csvfile)

    fig, axes = plt.subplots(nrows=2, ncols=3)
    fig.suptitle(csvfilename + ' plots', fontsize=16)
    fig.set_size_inches((18, 11))
    # nodes vs number clusters
    ax = df.plot(ax=axes[1, 0], kind='scatter', x=1, y=6, color='red')
    ax.set_xlabel('# of Nodes')
    ax.set_ylabel('# of clusters')
    axes[1, 0].set_title('# of nodes')

    # edges vs number clusters
    ax = df.plot(ax=axes[0, 0], kind='scatter', x=2, y=6, color='green')
    ax.set_xlabel('# of edges')
    ax.set_ylabel('# of clusters')
    axes[0, 0].set_title('# of edges')

    # density vs number clusters
    ax = df.plot(ax=axes[1, 1], kind='scatter', x=3, y=6, color='blue')
    ax.set_xlabel('Density')
    ax.set_ylabel('# of clusters')
    axes[1, 1].set_title('Density')

    # DegreeMean vs number clusters
    logger.debug(df.head())
    ax = df.plot(ax=axes[1, 2], kind='scatter', x=4, y=6, color='yellow')
    ax.set_xlabel('DegreeMean')
    ax.set_ylabel('# of clusters')
    axes[1, 2].set_title('DegreeMean')

    # number nodes vs number edges
    logger.debug(df.head())
    ax = df.plot(ax=axes[0, 2], kind='scatter', x=1, y=2, color='pink')
    ax.set_xlabel('# of nodes')
    ax.set_ylabel('# of edges')
    axes[0, 2].set_title('# nodes vs # edges')

    # Degree Stddev vs number clusters
    ax = df.plot(ax=axes[0, 1], kind='scatter', x=5, y=6, color='orange')
    ax.set_xlabel('DegreeStddev')
    ax.set_ylabel('# of clusters')
    axes[0, 1].set_title('Degree Stddev')

    # display figure 1 (the scatter plots) and dont pause (block=False)
    plt.show(block=False)

    # histogram plot using pandas feature df.hist that
    # creates a figure with 6 subplots (one for each column)
    hist = df.hist(column=[1, 2, 3, 4, 5, 6], figsize=(11, 8))
    hist[0][0].title.set_text('# nodes histogram')
    logger.debug(hist[0][0])
    hist[0][1].title.set_text('# edges histogram')
    hist[1][0].title.set_text('Density histogram')
    hist[1][1].title.set_text('Degree mean histogram')
    hist[2][0].title.set_text('Degree stddev histogram')
    hist[2][1].title.set_text('# clusters histogram')
    hfig = plt.figure(2)
    hfig.suptitle(csvfilename + ' histogram plots', fontsize=16)

    # display figure 2 (histograms) and wait (block=True)
    logger.info('Displaying figures and '
                'waiting for user to close figures')
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
