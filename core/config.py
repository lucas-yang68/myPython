# -*- coding: utf-8 -*-
from pathlib import Path
import yaml
import logging.config

_cfg = None   # 模块私有变量

def load_config(path: str | None = None):
    global _cfg
    if _cfg is not None:
        return _cfg

    if path is None:
        path = Path(__file__).resolve().parent.parent / "config" / "config_dev.yaml"

    with open(path, "r", encoding="utf-8") as f:
        _cfg = yaml.safe_load(f)

    return _cfg


def get_config():
    global _cfg
    if _cfg is None:
        path = 'D:\myPython\myPython\config\config_dev.yaml' 
        with open(path, "r", encoding="utf-8") as f:
            _cfg = yaml.safe_load(f)
        #raise RuntimeError("config not loaded, call load_config() first")
    return _cfg

def init_logging(path="config/logging.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logging.config.dictConfig(cfg)
    
init_logging('D:\myPython\myPython\config\logging.yaml')
