import yaml

def load_yaml_config(path):
    """Load YAML file and return as dict"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def flatten_config(cfg):
    """Flatten a 2-level config dict into top-level keys for argparse defaults"""
    flat = {}
    for section, section_values in cfg.items():
        if isinstance(section_values, dict):
            flat.update(section_values)
        else:
            flat[section] = section_values
    return flat