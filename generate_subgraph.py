#!/usr/bin/env python

import os
import sys
import json
import random
import argparse
import logging
import ndex2
from ndex2.nice_cx_network import DefaultNetworkXFactory
from ndex2.nice_cx_network import NiceCXNetwork
import networkx


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
    parser.add_argument('inputcx', help='Input CX file')
    parser.add_argument('outdir', help='Directory where output CX files'
                                       ' will be written.')
    parser.add_argument('--numnetworks', type=int, default=1,
                        help='Number of networks to generate')
    parser.add_argument('--numnodes', type=int,
                        help='# of nodes subgraph should have, if unset'
                             'code relies on --minnodes, --maxnodes and'
                             'randomly selects a value')
    parser.add_argument('--logconf', default=None,
                        help='Path to python logging configuration file in '
                             'this format: https://docs.python.org/3/library/'
                             'logging.config.html#logging-config-fileformat '
                             'Setting this overrides -v parameter which uses '
                             ' default logger. (default None)')
    parser.add_argument('--writeasedgelist', action='store_true',
                        help='If set, output edge lists')
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


class NiceCXNetworkFactory(object):
    """
    Factory to create NiceCXNetwork objects
    """
    def __init__(self, inputcxfile):
        """
        Constructor
        """
        self._inputcxfile = inputcxfile

    def get_nice_cx_network(self):
        """
        Gets :py:class:`~ndex2.nice_cx_network.NiceCXNetwork`
        generated from CX file passed into constructor of this object
        :return:
        :rtype: :py:class:`~ndex2.nice_cx_network.NiceCXNetwork`
        """
        return ndex2.create_nice_cx_from_file(self._inputcxfile)


class NiceCXNetworkXConvertor(object):
    """

    """
    def __init__(self):
        """
        Constructor
        """
        self._netxfac = DefaultNetworkXFactory()

    def get_networkx(self, network):
        """

        :param network:
        :type network: :py:class:`~ndex2.nice_cx_network.NiceCXNetwork`
        :return:
        """
        return self._netxfac.get_graph(network)


class NetworkxToNiceCXConverter(object):
    """

    """
    def __init__(self):
        """
        constructor
        """
        pass

    def get_nice_cx_network(self, netx_network):
        """

        :param netx_network:
        :return:
        """
        nice_cx = NiceCXNetwork()
        logger.info('Keys for graph: ' + str(netx_network.graph.keys()))
        for attr_name in netx_network.graph.keys():
            nice_cx.set_network_attribute(attr_name, values=netx_network.graph[attr_name])

        node_id_map = self._add_nodes(netx_network, nice_cx)
        self._add_edges(netx_network, nice_cx, node_id_map)

        return nice_cx

    def _add_nodes(self, netx_network, nice_cx):
        """
        Adds nodes
        :param netx_network:
        :param nice_cx:
        :return:
        """
        node_id_map = {}
        reserved_attrs = ['name', 'represents']
        for node in netx_network.nodes(data=True):
            # logger.info('node: ' + str(node))
            node_keys = node[1].keys()
            if 'name' in node_keys:
                node_name = node[1]['name']
            else:
                node_name = str(node[0])
            if 'represents' in node_keys:
                node_represents = node[1]['represents']
            else:
                node_represents = None
            node_id = nice_cx.create_node(node_name, node_represents)
            node_id_map[node[0]] = node_id
            for attr_name in node[1].keys():
                if attr_name in reserved_attrs:
                    continue
                nice_cx.add_node_attribute(node_id, attr_name, node[1][attr_name])
        return node_id_map

    def _add_edges(self, netx_network, nice_cx, node_id_map):
        """
        Adds edges
        :param netx_network:
        :param nice_cx:
        :return:
        """
        for edge in netx_network.edges(data=True):
            # logger.info('edge: ' + str(edge))
            source_node_id = node_id_map[edge[0]]
            target_node_id = node_id_map[edge[1]]
            edge_keys = edge[2].keys()
            if 'interaction' in edge_keys:
                interaction = edge[2]['interaction']
            else:
                interaction = None
            edge_id = nice_cx.create_edge(source_node_id, target_node_id, interaction)
            for attr_name in edge_keys:
                if attr_name == 'interaction':
                    continue
                nice_cx.add_edge_attribute(edge_id, attr_name, edge[2][attr_name])


class SubGraphGenerator(object):
    """
    Generates sub graphs
    """
    def __init__(self, netx_network, numnetworks=None, numnodes=None,
                 seed=None):
        """
        Constructor
        """
        self._netx = netx_network
        self._numnetworks = numnetworks
        self._numnodes = numnodes
        self._seed = seed

    def get_next_subgraph(self):
        """

        :param netx_network:
        :return:
        :rtype: :py:class:`~networkx.Graph`
        """
        nodelist = list(self._netx.nodes)
        counter = 0
        while counter < self._numnetworks:
            logger.info('Generating network # ' + str(counter))
            rand_nodes = random.choices(nodelist, k=self._numnodes)
            subgraph = self._netx.subgraph(rand_nodes)
            unfrozen_copy_of_subgraph = networkx.Graph(subgraph)
            unfrozen_copy_of_subgraph.remove_nodes_from(networkx.isolates(subgraph))
            if len(unfrozen_copy_of_subgraph) == 0:
                logger.debug('Network has no nodes. Skipping')
                continue
            counter += 1
            yield unfrozen_copy_of_subgraph


def run(theargs, nice_cx_fac=None,
        netxconvertor=None,
        nicecxconvertor=None):
    """
    Main flow of processing

    :param theargs:
    :return:
    """
    if nice_cx_fac is None:
        nice_cx_fac = NiceCXNetworkFactory(os.path.abspath(theargs.inputcx))

    if netxconvertor is None:
        netxconvertor = NiceCXNetworkXConvertor()

    if nicecxconvertor is None:
        nicecxconvertor = NetworkxToNiceCXConverter()

    nicecx = nice_cx_fac.get_nice_cx_network()
    netx_network = netxconvertor.get_networkx(nicecx)
    num_nodes = netx_network.number_of_nodes()
    graphgenerator = SubGraphGenerator(netx_network, numnetworks=theargs.numnetworks,
                                       numnodes=theargs.numnodes)
    counter = 1
    if not os.path.isdir(theargs.outdir):
        os.makedirs(theargs.outdir, mode=0o755)
    for netxsubnet in graphgenerator.get_next_subgraph():
        logger.info('Generated graph density: ' +
                    str(networkx.classes.function.density(netxsubnet)))

        nicecx = nicecxconvertor.get_nice_cx_network(netxsubnet)
        num_nodes = len(nicecx.get_nodes())
        nicecx.set_name(str(counter) + ' - ' + str(num_nodes) +
                        ' node subgraph of ' +
                        str(nicecx.get_name()))
        with open(os.path.join(theargs.outdir, str(counter) + '.cx'), 'w') as f:
            json.dump(nicecx.to_cx(), f)
        counter += 1


def main(args):
    """
    Main entry point for program
    :param args:
    :return:
    """
    desc = """
    
    Generates subgraphs from input graph

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
