import yaml
import time
from thought_lifecycle.thoughtpool import LoadThoughtseedNetwork, ThoughtPoolAssigner, ThoughtSproutOrchestrator, ThoughtsproutTracker, ThoughtSproutManager
from thought_lifecycle.config import FeatureConfig

def main():
    manager = ThoughtSproutManager('test_config.yaml', 'thoughtseed_network.pkl')
    manager.run()

if __name__ == "__main__":
    main()