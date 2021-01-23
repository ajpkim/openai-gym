from collections import namedtuple
import yaml

def load_config(config_file):
    with open(config_file, mode='r') as f:
        config = yaml.safe_load(f)
        config = namedtuple('Config', field_names=list(config.keys()))(**config)
    return config