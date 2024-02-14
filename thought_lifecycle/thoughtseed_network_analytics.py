import networkx as nx
import numpy as np
import math
import random
import community as community_louvain
import leidenalg
import igraph as ig
import pickle

from thought_lifecycle.config import ThoughtseedNetworkConfig
from thought_lifecycle.thoughtseed_network import ThoughtseedNetwork

# Perform network analytics and print communities
class ThoughtseedNetworkAnalytics:
    def __init__(self, thoughtseed_network):
        self.graph = thoughtseed_network.graph
        self.degree_centrality = {}
        self.pagerank = {}
        self.communities = None

    def calculate_degree_centrality(self):
        self.degree_centrality = nx.degree_centrality(self.graph)

    def calculate_pagerank(self):
        self.pagerank = nx.pagerank(self.graph, alpha = 0.85)

    def detect_communities(self, algorithm='louvain'):
        print(f"ThoughtseedNetworkAnalytics: type(self.graph) = {type(self.graph)}")

        if algorithm == 'louvain':
            print(type(self.graph))
            self.communities = community_louvain.best_partition(self.graph, resolution=1.125)

        elif algorithm == 'leiden':
            # Convert the NetworkX graph to an igraph graph for use with the leidenalg package
            ig_graph = ig.Graph.from_networkx(self.graph)

            # Use the Leiden algorithm to detect communities
            partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)

            # Convert the partition to a dictionary similar to what community_louvain.best_partition returns
            self.communities = {node: i for i, community in enumerate(partition) for node in community}

        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
    def perform_analysis_and_print_communities(self, algorithm='louvain'):
        self.calculate_degree_centrality()
        self.calculate_pagerank()
        self.detect_communities(algorithm)

        # Print communities
        print("Communities detected using the Louvain algorithm:")
        communities_dict = {}
        for node, community_id in self.communities.items():
            if community_id not in communities_dict:
                communities_dict[community_id] = []
            communities_dict[community_id].append(node)

        for community_id, nodes in communities_dict.items():
            print(f"Community {community_id}: {nodes}")

        print(f"Total communities: {len(communities_dict)}")

    def describe_community(self, community_nodes):
        # Create a subgraph for the community
        community = self.graph.subgraph(community_nodes)

        # Only describe communities of size 3 or more
        if community.number_of_nodes() < 3:
            return None

        # Calculate properties
        size = community.number_of_nodes()
        density = nx.density(community)
        diameter = nx.diameter(community)
        avg_path_length = nx.average_shortest_path_length(community)

        # Return properties
        return {
            'size': size,
            'density': density,
            'diameter': diameter,
            'average_path_length': avg_path_length,
        }

    def count_large_communities(self):
        # Get a list of all community IDs
        community_ids = set(self.communities.values())

        # Count the number of communities of size 3 or more
        large_communities = 0
        for community_id in community_ids:
            community_nodes = [node for node, id in self.communities.items() if id == community_id]
            if len(community_nodes) >= 3:
                large_communities += 1

        return large_communities
    
    @staticmethod
    def load_state(filename):
        # Load the ThoughtseedNetwork object from a file
        with open(filename, 'rb') as f:
            return pickle.load(f)
        
    def save_state(self, filename):
        # Save the state of the ThoughtseedNetworkAnalytics object
        with open(filename, 'wb') as f:
            pickle.dump(self, f)