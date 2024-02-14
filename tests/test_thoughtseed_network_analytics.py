import yaml
import unittest
import networkx as nx
from thought_lifecycle.thoughtseed_network import ThoughtseedNetwork
from thought_lifecycle.thoughtseed_network_analytics import ThoughtseedNetworkAnalytics
from thought_lifecycle.config import ThoughtseedNetworkConfig, FeatureConfig


class TestThoughtseedNetworkAnalytics(unittest.TestCase):
    def setUp(self):
        # Load your configuration from test_config.yaml
        with open('test_config.yaml', 'r') as f:
            config_data = yaml.safe_load(f)

        feature_config = FeatureConfig(config_data['FeatureConfig'])  # Initialize FeatureConfig with data from YAML
        network_config = ThoughtseedNetworkConfig(config_data['ThoughtseedNetworkConfig'], feature_config)  # Initialize ThoughtseedNetworkConfig with data from YAML
        self.network = ThoughtseedNetwork(network_config)
        self.analytics = ThoughtseedNetworkAnalytics(self.network)
        
    def test_calculate_degree_centrality(self):
        self.analytics.calculate_degree_centrality()
        self.assertIsInstance(self.analytics.degree_centrality, dict)

    def test_calculate_pagerank(self):
        self.analytics.calculate_pagerank()
        self.assertIsInstance(self.analytics.pagerank, dict)

    def test_detect_communities(self):
        self.analytics.detect_communities()
        self.assertIsInstance(self.analytics.communities, dict)

    def test_perform_analysis_and_print_communities(self):
        self.analytics.perform_analysis_and_print_communities()
        self.assertIsInstance(self.analytics.communities, dict)

    def test_describe_community(self):
        self.network.initialize()
        community_nodes = list(self.network.graph.nodes)[:3]
        community_description = self.analytics.describe_community(community_nodes)
        self.assertIsInstance(community_description, dict)

    def test_count_large_communities(self):
        self.network.initialize()
        self.analytics.detect_communities()
        large_communities = self.analytics.count_large_communities()
        self.assertIsInstance(large_communities, int)

if __name__ == '__main__':
    unittest.main()