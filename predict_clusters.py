#!/usr/bin/env python

from __future__ import print_function
import os
import sys
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


TRAIN = 'train'
PREDICT = 'predict'

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
    parser.add_argument('--mode', choices=[TRAIN, PREDICT])
    parser.add_argument('--logconf', default=None,
                        help='Path to python logging configuration file in '
                             'this format: https://docs.python.org/3/library/'
                             'logging.config.html#logging-config-fileformat '
                             'Setting this overrides -v parameter which uses '
                             ' default logger. (default None)')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum')
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


class TestNet(nn.Module):

    def __init__(self):
        super(TestNet, self).__init__()
        self.fc1 = nn.Linear(7, 12)
        self.fc2 = nn.Linear(12, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x


def predict(net):
    """

    :param net:
    :return:
    """
    pass


def train_net(theargs, net, trainloader, validloader):
    """

    :param theargs:
    :param net:
    :return:
    """

    loss_function = nn.SmoothL1Loss()
    optimizer = optim.SGD(net.parameters(), lr=theargs.learning_rate,
                          weight_decay=theargs.weight_decay,
                          momentum=theargs.momentum, nesterov=True)
    for epoch in range(1, theargs.num_epochs):
        train_loss = []
        valid_loss = []
        net.train()
        for data, target in trainloader:
            output = net(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

    ## evaluation part
    net.eval()
    for data, target in validloader:
        output = net(data)
        loss = loss_function(output, target)
        valid_loss.append(loss.item())
    logger.info('Epoch: ' + str(epoch) + ' Training Loss: ' +
                str(np.mean(train_loss)) + 'Valid Loss: ' + str(np.mean(valid_loss)))


class PredictClusterData(torch.utils.data.Dataset):
    """
    hi
    """
    def __init__(self):
        self._data = torch.randn(240, dtype=torch.float32).view(30, 8)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        target = self._data[index][-1]
        data_val = self._data[index][:-1]
        return data_val, target


def run(theargs):
    """
    Main flow of processing

    :param theargs:
    :return:
    """
    trainloader = torch.utils.data.DataLoader(PredictClusterData(), batch_size=10)
    validloader = torch.utils.data.DataLoader(PredictClusterData(), batch_size=10)
    net = TestNet()
    if theargs.mode == TRAIN:
        train_net(theargs, net, trainloader, validloader)
    return 0


def main(args):
    """
    Main entry point for program
    :param args:
    :return:
    """
    desc = """
    
    Cluster prediction

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