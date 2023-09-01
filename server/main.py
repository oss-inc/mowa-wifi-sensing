import logging
import argparse
from runner.sv_train import Trainer
from runner.sv_test import Tester
from runner.utils import CustomFormatter
from datetime import datetime

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(CustomFormatter())
    logger.addHandler(handler)

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('mode', type=str, default='all', help='train: only train, test: only test, all: train+test')

    args = parser.parse_args()

    # For wandb group-name
    now = str(datetime.now())

    # Supervised learning
    if args.mode in ['train', 'all']:
        logger.info('Start supervised learning')
        trainer = Trainer("config.yaml")
        trainer.train()
    elif args.mode in ['test', 'all']:
        tester = Tester("config.yaml")
        tester.test()
    else:
        logger.error('Wrong argument!')