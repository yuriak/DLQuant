# -*- coding:utf-8 -*-
import json
import os


class ConfigHandler(object):
    def __init__(self, config_path, default_config):
        self.config_path = config_path
        self.config_content = default_config
    
    def is_config_exist(self):
        return os.path.exists(self.config_path)
    
    def read_config(self):
        if not os.path.exists(self.config_path):
            return self.config_content
        with open(self.config_path, 'r+') as f:
            return json.loads(f.read())
    
    def write_config(self, config):
        with open(self.config_path, 'w+') as f:
            f.write(json.dumps(config))
        self.config_content=config
