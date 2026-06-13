"""
LNO-IonTransport Pipeline: Configuration Loader Utility
"""
import yaml
import os

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[-] Configuration file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"[+] Configuration successfully loaded from {config_path}")
    return config
