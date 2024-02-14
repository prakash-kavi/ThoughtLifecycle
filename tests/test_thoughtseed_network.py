import unittest
import networkx as nx
import yaml
from thought_lifecycle.thoughtseed_network import ThoughtseedNetwork
from thought_lifecycle.config import ThoughtseedNetworkConfig, FeatureConfig

class TestThoughtseedNetwork(unittest.TestCase):
    def setUp(self):
        with open('test_config.yaml', 'r') as f:
            config_data = yaml.safe_load(f)
        
        feature_config = FeatureConfig(config_data['FeatureConfig'])
        self.thoughtseed_network_config = ThoughtseedNetworkConfig(config_data['ThoughtseedNetworkConfig'], feature_config)
        self.thoughtseed_network = ThoughtseedNetwork(self.thoughtseed_network_config)

    def test_initialize(self):
        self.thoughtseed_network.initialize()
        self.assertEqual(len(self.thoughtseed_network.thoughtseeds), self.thoughtseed_network_config.num_thoughtseeds)
        self.assertEqual(self.thoughtseed_network.graph.number_of_nodes(), self.thoughtseed_network_config.num_thoughtseeds)
        expected_edges = self.thoughtseed_network_config.num_thoughtseeds * (self.thoughtseed_network_config.num_thoughtseeds - 1) // 2
        self.assertEqual(self.thoughtseed_network.graph.number_of_edges(), expected_edges)

    def test_graph_type(self):
        self.assertIsInstance(self.thoughtseed_network.graph, nx.Graph)

    def test_generate_thoughtseeds(self):
        self.thoughtseed_network.generate_thoughtseeds()
        self.assertEqual(len(self.thoughtseed_network.thoughtseeds), 1000)

    def test_add_nodes_to_graph(self):
        self.thoughtseed_network.generate_thoughtseeds()
        self.thoughtseed_network.add_nodes_to_graph()
        self.assertEqual(self.thoughtseed_network.graph.number_of_nodes(), 1000)

    def test_add_edges_to_graph(self):
        self.thoughtseed_network.generate_thoughtseeds()
        self.thoughtseed_network.add_nodes_to_graph()
        self.thoughtseed_network.add_edges_to_graph()
        # Assuming the graph is a complete graph, the number of edges should be n*(n-1)/2
        expected_edges = self.thoughtseed_network_config.num_thoughtseeds * (self.thoughtseed_network_config.num_thoughtseeds - 1) // 2
        self.assertEqual(self.thoughtseed_network.graph.number_of_edges(), expected_edges)

    def test_normalize_weights(self):
        self.thoughtseed_network.generate_thoughtseeds()
        self.thoughtseed_network.add_nodes_to_graph()
        self.thoughtseed_network.add_edges_to_graph()
        self.thoughtseed_network.normalize_weights()
        weights = [data['weight'] for u, v, data in self.thoughtseed_network.graph.edges(data=True)]
        self.assertTrue(all(0 <= weight <= 1 for weight in weights))  # All weights should be between 0 and 1 after normalization

if __name__ == '__main__':
    unittest.main()