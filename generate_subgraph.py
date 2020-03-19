#!/usr/bin/env python

import os
import sys
import json
import random
import argparse
import logging
import csv
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
    parser.add_argument('--gmtfile', help='If set, generate graphs'
                                          'using GO term genes. This '
                                          'assumes a file'
                                          'from sigdb like this one is '
                                          'used: https://www.gsea-msigdb'
                                          '.org/gsea/msigdb/download_file'
                                          '.jsp?filePath=/msigdb/release/'
                                          '7.0/c5.all.v7.0.symbols.gmt '
                                          'Note: currently this ignores'
                                          '--numnetworks and --numnodes '
                                          'parameters')
    parser.add_argument('--mingotermcutoff', type=int,
                        default=30,
                        help='Minimum number of nodes subgraph must '
                             'have to be generated. Only used '
                             'in go term mode ie --gmtfile set')
    parser.add_argument('--numnetworks', type=int, default=1,
                        help='Number of networks to generate')
    parser.add_argument('--numnodes', type=int,
                        help='# of nodes randomly picked from parent graph')
    parser.add_argument('--logconf', default=None,
                        help='Path to python logging configuration file in '
                             'this format: https://docs.python.org/3/library/'
                             'logging.config.html#logging-config-fileformat '
                             'Setting this overrides -v parameter which uses '
                             ' default logger. (default None)')
    parser.add_argument('--writecx', action='store_true',
                        help='If set, output cx')
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
    Generates sub graphs in generator function :py:func:`~get_next_subgraph~
    """
    def __init__(self, netx_network, numnetworks=None, numnodes=None):
        """
        Constructor

        :param netx_network: Network used to generate subgraphs
        :type netx_network: :py:class:`networkx.Graph`
        :param numnetworks: Number of networks to generate
        :type numnetworks: int
        :param numnodes:
        :type numnodes: int
        """
        self._netx = netx_network
        self._numnetworks = numnetworks
        self._numnodes = numnodes

    def get_next_subgraph(self):
        """
        Generator function that creates a subgraph from the `netx_network`
        passed in the constructor by randomly selecting nodes and creating
        a graph using original edges

        :return: subgraph of random nodes with edges from parent network passed
                 in constructor
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


class GoSubGraphGenerator(object):
    """
    Generates sub graphs from network passed into constructor
    that contain genes from the go terms passed in
    """
    def __init__(self, netx_network, gmtfile=None,
                 mincutoff=30):
        """
        Constructor

        :param netx_network: networkx network
        :param netx_network: :py:class:`networkx.Graph`
        :param gmtfile: GMT file of go terms in tsv format
                        <GO NAME> <URL> <GENE1> <GENE2>
        :param gmtfile: str
        :param mincutoff: only keep network if after isolate
                          removal there remains this many nodes
        :type mincutoff: int
        """
        self._netx = netx_network
        self._gmtfile = gmtfile
        self._mincutoff = mincutoff

    def _get_node_dict(self):
        """
        Examine network passed in constructor
        and build a :py:class:`dict` with key
        set to node 'name' uppercased and value
        set to id of node

        :return: dict of gene names as key and
                 node ids as values
        :rtype: dict
        """
        node_dict = {}
        for nodeid, node in self._netx.nodes(data=True):
            if 'name' not in node:
                continue
            node_dict[node['name'].upper()] = nodeid

        return node_dict

    def _get_list_of_nodeids_for_genes(self, node_dict, genelist):
        """
        Given a gene list return a list of node ids from `node_dict`
        that correspond to genes in `genelist`

        :param node_dict:
        :type node_dict: dict
        :param genelist:
        :type genelist: list
        :return: list of node ids that correspond to
                 genes passed in `genelist`
        :rtype: list
        """
        node_id_list = []
        for gene in genelist:
            uppercase_gene = gene.upper()
            if uppercase_gene in node_dict:
                node_id_list.append(node_dict[uppercase_gene])
        return node_id_list

    def get_next_subgraph(self):
        """
        Generator function that return subgraphs of original network
        with only nodes and edges for a given go term from
        the GMT file passed in the constructor.

        NOTE: any isolates (nodes without edges) are removed
        :param mincutoff: Minimum number of nodes that subgraph
                          that must exist to return it
        :type mincutoff: int
        :return: subgraph of network
        :rtype: :py:class:`networkx.Graph`
        """
        node_dict = self._get_node_dict()
        with open(self._gmtfile, 'r') as f:

            reader = csv.reader(f, delimiter='\t')
            try:
                for row in reader:
                    logger.info('Generating network ' + str(row[0]))
                    node_id_list = self.\
                        _get_list_of_nodeids_for_genes(node_dict,
                                                       row[2:])
                    subgraph = self._netx.subgraph(node_id_list)

                    unfrozen_copy_of_subgraph = networkx.Graph(subgraph)
                    unfrozen_copy_of_subgraph.\
                        remove_nodes_from(networkx.isolates(subgraph))
                    node_cnt = len(unfrozen_copy_of_subgraph)
                    if node_cnt < self._mincutoff:
                        logger.info(str(row[0]) + ' network has ' +
                                     str(node_cnt) + ' which is less then ' +
                                     str(self._mincutoff) + ' nodes. Skipping')
                        continue
                    unfrozen_copy_of_subgraph.graph['goterm'] = row[0]
                    yield unfrozen_copy_of_subgraph
            except csv.Error as e:
                logger.error('Error parsing {}, '
                             'line {}: {}'.format(self._gmtfile,
                                                  reader.line_num,
                                                  e))
                raise e


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
    if theargs.gmtfile is not None:
        graphgenerator = GoSubGraphGenerator(netx_network, gmtfile=theargs.gmtfile,
                                             mincutoff=theargs.mingotermcutoff)
    else:
        graphgenerator = SubGraphGenerator(netx_network, numnetworks=theargs.numnetworks,
                                           numnodes=theargs.numnodes)
    counter = 1
    if not os.path.isdir(theargs.outdir):
        os.makedirs(theargs.outdir, mode=0o755)
    for netxsubnet in graphgenerator.get_next_subgraph():
        logger.info('Generated graph density: ' +
                    str(networkx.classes.function.density(netxsubnet)))
        if theargs.writecx is True:
            _write_cx(nicecxconvertor, netxsubnet, theargs.outdir, counter)

        logger.info('Writing edge list ' + str(counter))
        edgelistfile = os.path.join(theargs.outdir,
                                    str(counter) + '.tsv')
        networkx.readwrite.edgelist.write_edgelist(netxsubnet,
                                                   edgelistfile,
                                                   delimiter='\t', data=False)
        counter += 1


def _write_cx(converter, netx_graph, outdir, counter):
    """

    :param self:
    :param outdir:
    :param counter:
    :param nicecx:
    :return:
    """
    nicecx = converter.get_nice_cx_network(netx_graph)
    num_nodes = len(nicecx.get_nodes())
    nicecx.set_name(str(counter) + ' - ' + str(num_nodes) +
                    ' node subgraph of ' +
                    str(nicecx.get_name()))
    with open(os.path.join(outdir, str(counter) + '.cx'), 'w') as f:
        json.dump(nicecx.to_cx(), f)


def main(args):
    """
    Main entry point for program
    :param args:
    :return:
    """
    desc = """
    
    Generates sub graphs from input graph via two modes 
    (Default & GO)
    
    
    Default mode:
    
    A number (via --numnetworks) of 
    sub graphs are generated by extracting a random number of selected
    nodes (via --numnodes) along with all corresponding edges from the
    input network.
    Any isolates (nodes no edges) are removed and due to this removal
    the number of ndoes might be less then --numnodes
    
    GO Term mode:
    
    Enabled by setting --gmtfile to a GO term TSV file 
    that can be downloaded here:
    
    https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/release/7.0/c5.all.v7.0.symbols.gmt
    
    In this mode a sub graph network is generated for each go term
    selecting nodes that corresponding to genes for that go term from the
    input network. The gene names are assumed to be on the 
    'name' attribute of the node. Any isolates (nodes no edges) 
    are removed. In output CX (written if --writecx added) 
    a network attribute 'goterm' is added to denote the 
    corresponding GO term

    TODO: Add support to also examine 'member' list attribute 
          with *.: prefix removed

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
