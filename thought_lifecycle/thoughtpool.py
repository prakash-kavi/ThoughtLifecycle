import numpy as np
import time
import random
from sortedcontainers import SortedSet
import pickle
import yaml
import sys
import os
import statistics

from thought_lifecycle.thoughtseed import Thoughtseed, ThoughtseedGenerator
from thought_lifecycle.thoughtseed_network import ThoughtseedNetwork
from thought_lifecycle.config import FeatureConfig, ThoughtseedNetworkConfig

class LoadThoughtseedNetwork:
    def __init__(self, filename):
        self.filename = filename
        self.thoughtseed_network = None
        self.num_activated = None
        self.num_not_activated = None

    def load_thoughtseed_network(self):
        # Add the parent directory to the Python path
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # Load the ThoughtseedNetwork object
        try:
            with open(self.filename, 'rb') as f:
                self.thoughtseed_network = pickle.load(f)
        except (FileNotFoundError, IOError):
            print(f"Error loading {self.filename}")
            return
    
    def get_thoughtseeds(self):
        # Return the thoughtseeds from the ThoughtseedNetwork object
        return self.thoughtseed_network.thoughtseeds if self.thoughtseed_network else None

class Thoughtsprout(Thoughtseed):
    def __init__(self, feature_values, memory_pattern, energy_level, energy_change):
        super().__init__(feature_values, memory_pattern)
        self.energy_level = self.calculate_energy_level(energy_level, energy_change)  # Calculate the energy level
        self.time_activated = time.time()  # Set the activation time to the current time

    def calculate_energy_level(self, energy_level, energy_change):
        # Calculate the energy level based on the energy level of the Thoughtseed and the energy change
        return energy_level + energy_change

class ThoughtSproutOrchestrator:
    def __init__(self, thoughtpool_assigner, thoughtsprout_tracker):
        self.thoughtpool_assigner = thoughtpool_assigner
        self.thoughtsprout_tracker = thoughtsprout_tracker

    def orchestrate(self, thoughtseeds):
        for thoughtseed in thoughtseeds:
            if thoughtseed.is_activated():
                # Create a Thoughtsprout from an active Thoughtseed
                thoughtsprout = Thoughtsprout(thoughtseed.feature_values, thoughtseed.memory_pattern, thoughtseed.energy_level, 0.1)
                
                # Assign the Thoughtsprout to a ThoughtPool
                pool_key = self.thoughtpool_assigner.assign_to_pool(thoughtsprout)
                
                # Assign energy level to the Thoughtsprout based on the assigned pool
                thoughtsprout.energy_level = self.calculate_energy_level(thoughtsprout.energy_level, pool_key)
                
                # Track the new thoughtsprout
                self.thoughtsprout_tracker.track(thoughtsprout)

    def calculate_energy_level(self, base_energy, pool_key):
        # Define energy adjustments for each pool
        energy_adjustments = {
            'positive': 0.1,
            'negative': -0.1,
            'neutral': 0,
            'positive_saliency': 0.2,
            'negative_saliency': -0.2
        }
        
        # Calculate the new energy level based on the base energy and the adjustment for the assigned pool
        return base_energy + energy_adjustments[pool_key]

class ThoughtPool:
    def __init__(self, valence):
        self.valence = valence  # Valence of the pool
        self.capacity = 7  # Maximum number of Thoughtsprouts in the pool
        self.thoughtsprouts = SortedSet(key=lambda sprout: sprout.energy_level)

    def add_sprout(self, sprout):
        if len(self.thoughtsprouts) < self.capacity:
            self.thoughtsprouts.add(sprout)
        else:
            self.thoughtsprouts.pop(0)
            self.thoughtsprouts.add(sprout)

class ThoughtPoolAssigner:
    def __init__(self, feature_config):
        # Fetch the mean and standard deviation of the valence from the feature configuration
        self.mean_valence = feature_config.features['Valence'].parameters['mu']
        self.std_dev_valence = feature_config.features['Valence'].parameters['sigma']
        
        # Initialize the thought pools for different valences and saliencies
        self.thought_pools = {
            'positive': [ThoughtPool('positive')],
            'negative': [ThoughtPool('negative')],
            'neutral': [ThoughtPool('neutral')],
            'positive_saliency': [ThoughtPool('positive_saliency')],
            'negative_saliency': [ThoughtPool('negative_saliency')]
        }

    def assign_to_pool(self, sprout):
        # Determine the valence of the sprout
        valence = sprout.feature_values['Valence']
        
        # Assign the sprout to a valence pool based on its valence
        if np.abs(valence - self.mean_valence) < self.std_dev_valence:
            valence_key = 'neutral'
        elif valence - self.mean_valence > self.std_dev_valence:
            valence_key = 'positive'
        else:
            valence_key = 'negative'

        # If the sprout's valence is more than 2 standard deviations away from the mean,
        # assign it to a saliency pool
        if np.abs(valence - self.mean_valence) > 2 * self.std_dev_valence:
            saliency_key = 'positive_saliency' if valence - self.mean_valence > 0 else 'negative_saliency'
            self.add_sprout_to_pool(sprout, saliency_key)
            return saliency_key
        else:
            # If the sprout's valence is within 2 standard deviations of the mean,
            # assign it to a valence pool
            self.add_sprout_to_pool(sprout, valence_key)
            return valence_key

    def add_sprout_to_pool(self, sprout, pool_key):
        # Add the sprout to the appropriate pool
        for pool in self.thought_pools[pool_key]:
            if len(pool.thoughtsprouts) < pool.capacity:
                pool.add_sprout(sprout)
                return

        new_pool = ThoughtPool(pool_key)
        new_pool.add_sprout(sprout)
        self.thought_pools[pool_key].append(new_pool)

    def print_status(self):
        for pool_key, pools in self.thought_pools.items():
            print(f"Status for {pool_key} pools:")
            print(f"Number of pools: {len(pools)}")
            
            pool_sizes = [len(pool.thoughtsprouts) for pool in pools]
            print(f"Average pool size: {sum(pool_sizes) / len(pool_sizes) if pool_sizes else 0}")
            print(f"Pool size standard deviation: {statistics.stdev(pool_sizes) if len(pool_sizes) > 1 else 0}")
            
            all_thoughtsprouts = [sprout for pool in pools for sprout in pool.thoughtsprouts]
            if all_thoughtsprouts:
                energies = [sprout.energy_level for sprout in all_thoughtsprouts]
                print(f"Energy stats: min={min(energies)}, max={max(energies)}, mean={sum(energies)/len(energies)}, std_dev={statistics.stdev(energies) if len(energies) > 1 else 0}")

class ThoughtsproutTracker:
    def __init__(self):
        self.thoughtsprouts = []  # Initialize an empty list to store thoughtsprouts
        self.timestamps = []  # Initialize an empty list to store timestamps

    def track(self, thoughtsprout):
        # Add the thoughtsprout to the list
        self.thoughtsprouts.append(thoughtsprout)
        # Add the activation time to the timestamps list
        self.timestamps.append(thoughtsprout.time_activated)

    def print_status(self):
        print(f"Total thoughtsprouts tracked: {len(self.thoughtsprouts)}")
        timestamps = [sprout.time_activated for sprout in self.thoughtsprouts]
        print(f"Timestamp stats: min={min(timestamps)}, max={max(timestamps)}, mean={sum(timestamps)/len(timestamps)}")

class ThoughtSproutManager:
    def __init__(self, config_filename, thoughtseed_network_filename):
        self.config_filename = config_filename
        self.thoughtseed_network_filename = thoughtseed_network_filename
        self.feature_config = None
        self.thoughtseeds = None
        self.thoughtpool_assigner = None
        self.thoughtsprout_tracker = None

    def load_config_and_thoughtseeds(self):
        # Load the configuration from the YAML file
        with open(self.config_filename, 'r') as f:
            config_data = yaml.safe_load(f)

        # Create a FeatureConfig object from the configuration data
        self.feature_config = FeatureConfig(config_data['FeatureConfig'])

        # Load the ThoughtseedNetwork object
        loader = LoadThoughtseedNetwork(self.thoughtseed_network_filename)
        loader.load_thoughtseed_network()

        # Get the thoughtseeds from the loader
        self.thoughtseeds = loader.get_thoughtseeds()

    def orchestrate_thoughtsprouts(self):
        # Create a ThoughtPoolAssigner
        self.thoughtpool_assigner = ThoughtPoolAssigner(self.feature_config)

        # Create a ThoughtsproutTracker
        self.thoughtsprout_tracker = ThoughtsproutTracker()

        # Create a ThoughtSproutOrchestrator
        thoughtsprout_orchestrator = ThoughtSproutOrchestrator(self.thoughtpool_assigner, self.thoughtsprout_tracker)

        # Orchestrate the activation of thoughtseeds and their assignment to a thoughtpool
        thoughtsprout_orchestrator.orchestrate(self.thoughtseeds)

    def print_status(self):
        self.thoughtpool_assigner.print_status()
        self.thoughtsprout_tracker.print_status()

    def run(self):
        self.load_config_and_thoughtseeds()
        self.orchestrate_thoughtsprouts()
        self.print_status()

if __name__ == "__main__":
    manager = ThoughtSproutManager('test_config.yaml', 'thoughtseed_network.pkl')
    manager.run()
