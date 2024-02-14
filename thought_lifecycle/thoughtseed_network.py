import networkx as nx
import numpy as np
import math
import random
import pickle

from thought_lifecycle.thoughtseed import ThoughtseedGenerator
from thought_lifecycle.config import FeatureConfig, ThoughtseedNetworkConfig

# Intialize the Thoughtseed Network to store and manage thoughtseeds
class ThoughtseedNetwork:
    def __init__(self, config):
        self.config = config
        self.graph = nx.Graph()
        self.thoughtseeds = []

    def generate_thoughtseeds(self):
        # Generate thoughtseeds based on the network configuration, containing the thoughtseed number and feature configurations of thoughtseed 
        generator = ThoughtseedGenerator(self.config)
        self.thoughtseeds = generator.generate_thoughtseeds()

    def add_nodes_to_graph(self):
        # Add each thoughtseed as a node to the graph
        for i, thoughtseed in enumerate(self.thoughtseeds):
            self.graph.add_node(i, feature_values=thoughtseed.feature_values, memory_pattern=thoughtseed.memory_pattern)

    def assign_edge_weights(self, ts1, ts2):
        # Assigns weights to an edge based on the valence and complexity values of the two Thoughtseeds it connects.
        valence1 = ts1.feature_values['Valence']
        valence2 = ts2.feature_values['Valence']
        complexity1 = ts1.feature_values['Complexity']
        complexity2 = ts2.feature_values['Complexity']

        # Get the mean and standard deviation of the valence from the feature_config
        mean_valence = self.config.feature_config.features['Valence'].parameters['mu']
        std_dev_valence = self.config.feature_config.features['Valence'].parameters['sigma']

        # Calculate the absolute difference from the mean valence
        diff_from_mean1 = abs(valence1 - mean_valence)
        diff_from_mean2 = abs(valence2 - mean_valence)

        # Assign a base weight based on the valence values
        if diff_from_mean1 > 2 * std_dev_valence and diff_from_mean2 > 2 * std_dev_valence:
            valence_weight = 16 if (valence1 - mean_valence) * (valence2 - mean_valence) > 0 else 0
        elif (diff_from_mean1 > 2 * std_dev_valence and diff_from_mean2 > std_dev_valence) or (diff_from_mean2 > 2 * std_dev_valence and diff_from_mean1 > std_dev_valence):
            valence_weight = 12 if (valence1 - mean_valence) * (valence2 - mean_valence) > 0 else 1
        elif diff_from_mean1 > std_dev_valence or diff_from_mean2 > std_dev_valence:
            valence_weight = 5
        else:
            valence_weight = 1  # Default value

        # Assign a weight based on the maximum complexity of the two thoughtseeds. Higher complexities result in higher weights.
        complexity_weight = max(complexity1, complexity2) ** 2

        return valence_weight, complexity_weight

    def add_edges_to_graph(self):
        for i in range(len(self.thoughtseeds)):
            for j in range(i+1, len(self.thoughtseeds)):
                # Calculate the edge weights based on the thoughtseeds' features
                valence_weight, complexity_weight = self.assign_edge_weights(self.thoughtseeds[i], self.thoughtseeds[j])
                # Add an edge between the thoughtseeds to the graph, with the calculated weights as edge attributes
                self.graph.add_edge(i, j, valence_weight=valence_weight, complexity_weight=complexity_weight)

    def normalize_weights(self):
        # Get the maximum and minimum weights for normalization
        max_valence_weight = max((data['valence_weight'] for u, v, data in self.graph.edges(data=True)), default=0)
        min_valence_weight = min((data['valence_weight'] for u, v, data in self.graph.edges(data=True)), default=0)
        max_complexity_weight = max((data['complexity_weight'] for u, v, data in self.graph.edges(data=True)), default=0)
        min_complexity_weight = min((data['complexity_weight'] for u, v, data in self.graph.edges(data=True)), default=0)

        # Normalize the weights and combine them with the given weights for complexity:0.3 and valence:0.7
        for u, v, data in self.graph.edges(data=True):
            normalized_valence_weight = ((data['valence_weight'] - min_valence_weight) / (max_valence_weight - min_valence_weight)) if max_valence_weight != min_valence_weight else 0
            normalized_complexity_weight = ((data['complexity_weight'] - min_complexity_weight) / (max_complexity_weight - min_complexity_weight)) if max_complexity_weight != min_complexity_weight else 0
            data['weight'] = 0.7 * normalized_valence_weight + 0.3 * normalized_complexity_weight

    def initialize(self):
        #Initializes the memory network by performing the following steps:
        self.generate_thoughtseeds()  # Generate thoughtseeds based on the configuration
        self.add_nodes_to_graph()     # Add each thoughtseed as a node in the graph
        self.add_edges_to_graph()     # Add edges between thoughtseeds in the graph based on their similarity
        self.normalize_weights()      # Normalize the weights of the edges in the graph

    def save_state(self,filename):
        # Save the state of the ThoughtseedNetwork object
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_state(filename):
        # Load the state of a ThoughtseedNetwork object from a file
        with open(filename, 'rb') as f:
            return pickle.load(f)