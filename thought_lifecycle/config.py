import yaml

class FeatureConfig:
    def __init__(self, config):
        # Initialize feature configuration of a thoughtseed with their distributions, parameters, and weights
        self.features = {name: self.Feature(feature['distribution'], feature['parameters'], feature.get('weight'))
                         for name, feature in config.items()}

    class Feature:
        def __init__(self, distribution, parameters, weight=None):
            # Initialize thoughtseed feature with distribution, parameters, and weight
            self.distribution = distribution
            self.parameters = parameters
            self.weight = weight

class ThoughtseedNetworkConfig:
    def __init__(self, network_config, feature_config):
        # Initialize thoughtseed network configuration
        self.num_thoughtseeds = network_config['num_thoughtseeds']
        self.global_energy_value = network_config['global_energy_value']
        self.feature_config = feature_config