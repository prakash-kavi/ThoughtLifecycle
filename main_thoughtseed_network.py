import random
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
import os

from thought_lifecycle.thoughtseed_network import ThoughtseedNetwork
from thought_lifecycle.thoughtseed_network_analytics import ThoughtseedNetworkAnalytics
from thought_lifecycle.config import FeatureConfig, ThoughtseedNetworkConfig

def main():
    # Load the configuration file
    with open('test_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Memory network configuration
    feature_config = FeatureConfig(config['FeatureConfig'])
    thoughtseed_network_config = ThoughtseedNetworkConfig(config['ThoughtseedNetworkConfig'], feature_config)

    # Initialize the memory network
    thoughtseed_network = ThoughtseedNetwork(thoughtseed_network_config)
    thoughtseed_network.initialize()

    # Print 20 thoughtseeds chosen randomly
    random_thoughtseeds = random.sample(thoughtseed_network.thoughtseeds, 20)
    for i, thoughtseed in enumerate(random_thoughtseeds):
        print(f"Thoughtseed {i+1}:")
        print(f"Feature values: {thoughtseed.feature_values}")
        print(f"Energy level: {thoughtseed.energy_level}")
        location, time_slot, activity, emotion = thoughtseed.decode_memory_pattern()
        print(f"Memory pattern: {' '.join(thoughtseed.memory_pattern[i:i+8] for i in range(0, len(thoughtseed.memory_pattern), 8))}")
        print(f"Parameters: Complexity={activity}, Valence={emotion}, Location={location}, Time Slot={time_slot}")

    # Perform network analytics and print communities
    network_analytics = ThoughtseedNetworkAnalytics(thoughtseed_network)
    network_analytics.perform_analysis_and_print_communities(algorithm='louvain')

    # Describe communities of size 3 or more
    community_ids = set(network_analytics.communities.values())
    for community_id in community_ids:
        community_nodes = [node for node, id in network_analytics.communities.items() if id == community_id]
        community_properties = network_analytics.describe_community(community_nodes)
        if community_properties is not None:
            print(f"Community {community_id}: {community_properties}")

    # Count the number of communities of size 3 or more
    large_communities = network_analytics.count_large_communities()
    print(f"Number of communities of size 3 or more: {large_communities}")

    # Plot the distribution of energy levels
    energy_levels = [thoughtseed.energy_level for thoughtseed in thoughtseed_network.thoughtseeds]

    sns.histplot(energy_levels, bins=10, kde=True)
    plt.title('Distribution of Energy Levels')
    plt.xlabel('Energy Level')
    plt.ylabel('Density')
    plt.show()

    # Save the state of the ThoughtseedNetwork object
    thoughtseed_network.save_state('thoughtseed_network.pkl')

    # Save the state of the ThoughtseedNetworkAnalytics object
    network_analytics.save_state('thoughtseed_network_analytics.pkl')

if __name__ == "__main__":
    main()