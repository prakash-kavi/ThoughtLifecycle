import numpy as np
from scipy.stats import truncnorm
from collections import OrderedDict

from thought_lifecycle.config import FeatureConfig

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Thoughtseed:
    # Thoughtseed is a memory construct. Using the axiom, "A thought is a thing, and thoughtseed is the seed of thought."
    # It can represent a single neuronal population or multile neuronal poulations in the memory network. We only have  aminimalist version here.
    def __init__(self, feature_values, memory_pattern):
        # Initialize Thoughtseed with feature values, memory pattern, energy level, and activation status
        self.feature_values = feature_values
        self.memory_pattern = memory_pattern
        self.energy_level = 0.5
        self.activation_status = False

    def receive_synapse(self, energy_input):
        # Increase energy level and check activation status.
        self.energy_level += energy_input
        self.is_activated()

    def is_activated(self, threshold=0.8):
        # Check if activation level exceeds threshold
        self.activation_status = sigmoid(self.energy_level) > threshold
        return self.activation_status

    def calculate_firing_cost(self, thoughtseed, feature_config):
        # Calculate cost of firing a thoughtseed
        noise = np.random.normal(0, 0.1)
        complexity_weight = feature_config.features['Complexity'].weight
        manifestation_strength_weight = feature_config.features['Manifestation Strength'].weight
        activation_energy = thoughtseed.feature_values['Activation Threshold']
        complexity_factor = 1 + thoughtseed.feature_values['Complexity']
        firing_cost = complexity_factor * activation_energy * np.exp(complexity_weight * thoughtseed.feature_values['Complexity'] + manifestation_strength_weight * thoughtseed.feature_values['Manifestation Strength']) + noise
        return firing_cost

    def decode_memory_pattern(self):
        # Interpret memory pattern into four 8-bit segments
        activity = int(self.memory_pattern[:8], 2)
        emotion = int(self.memory_pattern[8:16], 2)
        location = int(self.memory_pattern[16:24], 2)
        time_slot = int(self.memory_pattern[24:], 2)
        return location, time_slot, activity, emotion

class ThoughtseedGenerator:
    def __init__(self, thoughtseed_network_config):
        # Initialize generator with number of thoughtseeds and feature configuration
        self.num_thoughtseeds = thoughtseed_network_config.num_thoughtseeds
        self.feature_config = thoughtseed_network_config.feature_config

    def generate_feature_values(self):
        # Generate feature values based on their distributions
        feature_values = OrderedDict()
        for feature, config in self.feature_config.features.items():
            if config.distribution == 'normal':
                value = np.random.normal(config.parameters['mu'], config.parameters['sigma'])
            elif config.distribution == 'beta':
                value = np.random.beta(config.parameters['alpha'], config.parameters['beta'])
            elif config.distribution == 'truncated_normal':
                value = truncnorm.rvs((config.parameters['low'] - config.parameters['mu']) / config.parameters['sigma'], 
                                    (config.parameters['up'] - config.parameters['mu']) / config.parameters['sigma'], 
                                    loc=config.parameters['mu'], scale=config.parameters['sigma'])
            feature_values[feature] = np.round(value, 4)
        return feature_values

    def generate_memory_pattern(self, feature_values):
        # Generate random 8-bit segments for memory pattern
        memory_pattern = ''         
        normalized_value = int(255 * feature_values['Complexity'])
        memory_pattern += format(normalized_value, '08b')
        normalized_value = int(255 * feature_values['Valence'])
        memory_pattern += format(normalized_value, '08b')
        memory_pattern += format(np.random.randint(0, 256), '08b')
        memory_pattern += format(np.random.randint(0, 256), '08b')
        return memory_pattern

    def generate_thoughtseeds(self):
        # Generate thoughtseeds and calculate their energy levels
        thoughtseeds = []
        for _ in range(self.num_thoughtseeds):
            try:
                feature_values = self.generate_feature_values()
                memory_pattern = self.generate_memory_pattern(feature_values)
                thoughtseed = Thoughtseed(feature_values, memory_pattern)
                thoughtseeds.append(thoughtseed)
            except ValueError as e:
                print(f"Failed to generate thoughtseed: {e}")
        for thoughtseed in thoughtseeds:
            thoughtseed.energy_level += thoughtseed.calculate_firing_cost(thoughtseed, self.feature_config)
        return thoughtseeds