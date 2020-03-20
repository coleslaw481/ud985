#!/usr/bin/env python

from __future__ import print_function
import os
import sys
import argparse
import logging
import datetime
import numpy as np
import pandas
import torch
import torch.nn as nn
import torch.optim as optim


TRAIN = 'train'
PREDICT = 'predict'

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
    parser.add_argument('--mode', choices=[TRAIN, PREDICT], required=True,
                        help='Denotes which mode to run in')
    parser.add_argument('--model',
                        help='The model to use for prediction')
    defaultsave = 'predictclusters.' +\
                  str(datetime.datetime.now().timestamp()) +\
                  '.pt'
    parser.add_argument('--save', default=defaultsave,
                        help='The file to save the trained model to when'
                             '--mode is ' + TRAIN)
    parser.add_argument('--traindata', help='Path to training '
                                            'dataset (only needed if '
                                            '<mode> is ' + TRAIN + ')')
    parser.add_argument('--validationdata',
                        help='Path to validation '
                             'dataset (only needed if '
                             '<mode> is ' + TRAIN + ')')
    parser.add_argument('--logconf', default=None,
                        help='Path to python logging configuration file in '
                             'this format: https://docs.python.org/3/library/'
                             'logging.config.html#logging-config-fileformat '
                             'Setting this overrides -v parameter which uses '
                             ' default logger.')
    parser.add_argument('--plotgraphs', action='store_true',
                        help='If set, uses matplotlib to plot graphs')
    parser.add_argument('--matplotlibgui', default='Qt4Agg',
                        help='Library to use for plotting')
    parser.add_argument('--usecpu', action='store_true',
                        help='If set, use CPU instead of '
                             'first visible GPU')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number epochs')
    parser.add_argument('--batchsize', type=int, default=16,
                        help='Batch size')
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
                             '-vvvv = DEBUG, -vvvvv = NOTSET')

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


class PredictClustersNeuralNetwork(nn.Module):
    """
    Predict Clusters neural network which takes in
    four attributes about a network and passes those attributes
    to an eight node hidden layer and then a four layer hidden
    layer before outputting a single value ranged between 0 and 1
    which denotes number of clusters by taking that value and
    multipling by number of nodes
    """

    def __init__(self):
        """
        Constructor that sets up the 3 layers 4 node input,
        8 node hidden, and 4 node hidden
        """
        super(PredictClustersNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, 1)

    def forward(self, x):
        """
        Defines how network is linked up. Currently
        relu is used as activation function
        :param x:
        :return:
        """
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x


def predict(net):
    """

    :param net:
    :return:
    """
    pass


def get_device(theargs):
    """
    Examines theargs.usecpu if set then device is cpu
    otherwise it is in the first available GPU device
    if CUDA is available
    :param theargs:
    :return:
    """
    if theargs.usecpu:
        return torch.device('cpu')

    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train_net(theargs, net, trainloader, validloader):
    """
    Trains network

    :param theargs: command line arguments from :py:class:`Argparse`
    :param net: network to use for training
    :type net: :py:class:`torch.nn.Module`
    :param trainloader: training dataloader
    :type trainloader: :py:class:`torch.utils.data.DataLoader`
    :param validloader: validation dataloader
    :type validloader: :py:class:`torch.utils.data.DataLoader`
    :return: None
    """
    device = get_device(theargs)
    net.to(device)
    loss_function = nn.SmoothL1Loss()
    optimizer = optim.SGD(net.parameters(), lr=theargs.learning_rate,
                          weight_decay=theargs.weight_decay,
                          momentum=theargs.momentum, nesterov=True)
    epoch_losses = []
    valid_losses = []
    for epoch in range(1, theargs.num_epochs):
        train_loss = []
        valid_loss = []
        if len(epoch_losses) == 0:
            tl = 'NA'
        else:
            tl = epoch_losses[-1]
        logger.debug('Running epoch: ' +
                     str(epoch) + ' of ' +
                     str(theargs.num_epochs) + ' train loss: ' + str(tl))

        # training phase
        net.train()
        for data_raw, target_raw in trainloader:
            data = data_raw.to(device)
            target = target_raw.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        epoch_losses.append(np.mean(train_loss))

        # evaluation phase
        net.eval()
        for data, target in validloader:
            output = net(data)
            loss = loss_function(output, target)
            valid_loss.append(loss.item())

        valid_losses.append(np.mean(valid_loss))

    # plot training/validation loss
    if theargs.plotgraphs:
        plotgraphs(theargs, epoch_losses, valid_losses)

    save_trained_model(theargs, epoch=theargs.num_epochs,
                       model=net, optimizer=optimizer, loss=train_loss[-1])


def save_trained_model(theargs, epoch=None, model=None,
                       optimizer=None, loss=None):
    """
    Saves model to file specified in theargs.save
    :param epoch:
    :param model:
    :param optimizer:
    :param loss:
    :return:
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, theargs.save)


def plotgraphs(theargs, epoch_losses, valid_losses):
    """

    :param epoch_losses:
    :param valid_losses:
    :return:
    """
    import matplotlib
    matplotlib.use(theargs.matplotlibgui)
    import matplotlib.pyplot as plt
    plt.plot(epoch_losses, color='green', marker='o', linestyle='solid')
    plt.plot(valid_losses, color='red', marker='x', linestyle='solid')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Green is train loss and red is validation loss')
    plt.show()


class PredictClusterData(torch.utils.data.Dataset):
    """
    Creates a normalized predict cluster dataset
    """

    def __init__(self, inputfile):
        """
        Constructor that processes `inputfile` data into normalized
        dataset

        :param inputfile: Path to CSV file containing predictcluster data
        :type inputfile: str
        """
        df = self._load_data(inputfile)

        self._normalize_data(df)

        self._data = torch.Tensor(np.array(df))

    def _load_data(self, inputfile):
        """
        Loads the data
        :param inputfile:
        :return: pandas data frame
        """
        return pandas.read_csv(inputfile, delimiter=',', header=None)

    def _normalize_data(self, df):
        """
        normalizes data by making the following changes to the data frame (df)
        passed in. These changes are made in place

        The first column (usually the tsv file name) is dropped

        df[2] aka edges is set to # edges/# nodes ^2

        df[6] aka # clusters is set to # clusters / # nodes

        df[4] aka degree mean is set to degree mean / # nodes

        df[1] # nodes is removed
        :param df:
        :return:
        """
        del df[0]
        df.reset_index(drop=True, inplace=True)

        # set edges to num edges / num nodes^2
        df[2] = (df[2] / (df[1] * df[1]))

        # set number of clusters to num clusters / num nodes
        df[6] = (df[6] / df[1])

        # set degree mean to degree mean / num nodes
        df[4] = (df[4] / df[1])

        del df[1]
        df.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        target = self._data[index][-1].view(1)
        data_val = self._data[index][:-1].view(4)
        return data_val, target


def run(theargs):
    """
    Main flow of processing

    :param theargs:
    :return:
    """
    trainloader = torch.utils.data.DataLoader(PredictClusterData(theargs.traindata),
                                              batch_size=theargs.batchsize)
    validloader = torch.utils.data.DataLoader(PredictClusterData(theargs.validationdata),
                                              batch_size=theargs.batchsize)
    net = PredictClustersNeuralNetwork()
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
