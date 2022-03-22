import sys
import importlib
import argparse
from types import SimpleNamespace

# import config 
sys.path.append("config")
parser = argparse.ArgumentParser(description='')
parser.add_argument("-c", "--config", help="config filename")
parser_args, _ = parser.parse_known_args(sys.argv)
print("Using config file", parser_args.config)
args = importlib.import_module(parser_args.config).args
args["experiment_name"] = parser_args.config
args =  SimpleNamespace(**args)