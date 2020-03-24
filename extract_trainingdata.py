#!/usr/bin/env python

import os
import sys
import argparse
import logging
import pandas

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
    parser.add_argument('csvfile', help='Input hierarchy stats CSV')
    parser.add_argument('out', help='Directory where output CSV '
                                    'results will be stored')
    parser.add_argument('--numdatapoints', type=int, default=50,
                        help='Number of training data '
                             'points per cluster count')
    parser.add_argument('--validpercent', type=float, default=0.05)
    parser.add_argument('--testpercent', type=float, default=0.05)
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


class CreateNormalizedTrainingDataSets(object):
    """
    Creates normalized training datasets
    """
    def __init__(self, numdatapoints=50, testpercent=0.05,
                 validpercent=0.05):
        """
        Constructor
        """
        self._numdatapoints = numdatapoints
        self._testpercent = testpercent
        self._validpercent = validpercent

    def _get_validcluster_counts(self, input_dataframe):
        """

        :param input_dataframe:
        :return:
        """
        res = input_dataframe[6].value_counts(ascending=False)
        validclustercounts = []
        for entry in res.iteritems():
            if entry[1] < self._numdatapoints:
                break
            validclustercounts.append(entry[0])
        return validclustercounts

    def get_training_datasets(self, input_dataframe):
        """

        :param input_dataframe:
        :return: (training, validation, test)
        """
        normalized_df = None
        validclustercounts = self._get_validcluster_counts(input_dataframe)
        for count in validclustercounts:
            count_df = input_dataframe.loc[input_dataframe[6] == count]

            sample_df = count_df.sample(n=self._numdatapoints)
            if normalized_df is None:
                normalized_df = sample_df
            else:
                normalized_df = normalized_df.append(sample_df, ignore_index=True)

        all_df = normalized_df.sample(frac=1).reset_index(drop=True)
        num_rows = len(all_df)
        logger.debug('Total rows: ' + str(num_rows))
        train_percent = 1.0 - (self._validpercent + self._testpercent)
        train_rows_idx = round(num_rows*train_percent)
        logger.debug('Train rows index: ' + str(train_rows_idx))
        valid_rows_idx = round(num_rows*self._validpercent) + train_rows_idx
        logger.debug('Validation rows index: ' + str(valid_rows_idx))
        return all_df[0:train_rows_idx],\
               all_df[train_rows_idx:valid_rows_idx],\
               all_df[valid_rows_idx:]


def run(theargs):
    """
    Main flow of processing

    :param theargs:
    :return:
    """
    tdatasetmaker = CreateNormalizedTrainingDataSets(numdatapoints=theargs.numdatapoints,
                                                     testpercent=theargs.testpercent,
                                                     validpercent=theargs.validpercent)
    df = pandas.read_csv(theargs.csvfile, header=None)
    (train_df, valid_df, test_df) = tdatasetmaker.get_training_datasets(df)

    if not os.path.isdir(theargs.out):
        os.makedirs(theargs.out, mode=0o755)

    train_df.to_csv(os.path.join(theargs.out, 'train.csv'), index=False, header=False)
    valid_df.to_csv(os.path.join(theargs.out, 'valid.csv'), index=False, header=False)
    test_df.to_csv(os.path.join(theargs.out, 'test.csv'), index=False, header=False)
    return 0


def main(args):
    """
    Main entry point for program
    :param args:
    :return:
    """
    desc = """
    
    Takes aggregated CSV data from get_hierarchystats.py and
    attempts to create a training, validation, and testing
    datasets where there is an even distribution of clusters. 
    This is done by creating a histogram of clusters 
    (last column in CSV) and then randomly removing data from bins
    so they are roughly even.

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
