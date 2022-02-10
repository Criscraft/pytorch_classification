import ptutils.PytorchHelpers as ph
from ptutils.start_task import start_task
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str,
                        help="path to configs file")
    parser.add_argument("-g", "--gpu", type=int, default=0,
                        help="index of gpu to use")
    parser.add_argument("-s", "--seed", type=int, default=42,
                        help="seed to use")
    params = parser.parse_args()

    config_path = params.config_path
    config_module = ph.get_module(config_path)
    config = config_module.get_config(gpu=params.gpu, seed=params.seed)
    start_task(config, config_path)


if __name__ == '__main__':
    main()
