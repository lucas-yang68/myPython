# -*- coding: utf-8 -*-
import yaml
import logging.config
from core.config import load_config

def init_logging(path="config/logging.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logging.config.dictConfig(cfg)
    
load_config()
init_logging()

