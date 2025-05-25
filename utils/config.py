import yaml
from types import SimpleNamespace

def dict_to_namespace(d):
    ns = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, dict_to_namespace(v))
        else:
            setattr(ns, k, v)
    return ns

def load_config(path):
    with open(path) as f:
        cfg_dict = yaml.safe_load(f)
    return dict_to_namespace(cfg_dict)
