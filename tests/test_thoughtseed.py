import yaml
import unittest
from thought_lifecycle.thoughtseed import Thoughtseed, sigmoid
from thought_lifecycle.config import FeatureConfig

class TestThoughtseed(unittest.TestCase):
    def setUp(self):
        self.feature_values = {'Complexity': 0.5, 'Valence': 0.5, 'Manifestation Strength': 0.5, 'Activation Threshold': 0.5}
        self.memory_pattern = '00000000000000000000000000000000'
        self.thoughtseed = Thoughtseed(self.feature_values, self.memory_pattern)

    def test_sigmoid(self):
        self.assertEqual(sigmoid(0), 0.5)
        self.assertGreater(sigmoid(1), 0.5)
        self.assertLess(sigmoid(-1), 0.5)

    def test_initialization(self):
        self.assertEqual(self.thoughtseed.feature_values, self.feature_values)
        self.assertEqual(self.thoughtseed.memory_pattern, self.memory_pattern)
        self.assertEqual(self.thoughtseed.energy_level, 0.5)
        self.assertEqual(self.thoughtseed.activation_status, False)

    def test_receive_synapse(self):
        self.thoughtseed.receive_synapse(0.5)
        self.assertEqual(self.thoughtseed.energy_level, 1.0)

    def test_is_activated(self):
        self.assertFalse(self.thoughtseed.is_activated())
        self.thoughtseed.receive_synapse(1.0)  # Increase the synapse strength to ensure activation
        self.assertTrue(self.thoughtseed.is_activated())

    def test_calculate_firing_cost(self):
        with open('test_config.yaml', 'r') as f:
            config_data = yaml.safe_load(f)
        feature_config = FeatureConfig(config_data['FeatureConfig'])
        cost = self.thoughtseed.calculate_firing_cost(self.thoughtseed, feature_config)
        self.assertGreater(cost, 0)

    def test_decode_memory_pattern(self):
        location, time_slot, activity, emotion = self.thoughtseed.decode_memory_pattern()
        self.assertEqual(location, 0)
        self.assertEqual(time_slot, 0)
        self.assertEqual(activity, 0)
        self.assertEqual(emotion, 0)

if __name__ == '__main__':
    unittest.main()