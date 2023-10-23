import logging
import argparse
from runner.meta_train import Trainer_FSL
from runner.meta_test import Tester_FSL

from runner.sv_train import Trainer_SVL
from runner.sv_test import Tester_SVL
from runner.utils import CustomFormatter

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(CustomFormatter())
    logger.addHandler(handler)

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--learning', type=str, choices=['SVL', 'FSL'], required=True, help = 'Select Superviesd Learning(SVL) or Few-shot Learning(FSL)')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help= "Select train or test")

    args = parser.parse_args()
    
    if args.learning == 'SVL':
        if args.mode == 'train':
            logger.info('Start supervised learning')
            trainer = Trainer_SVL("config.yaml")
            trainer.train()

        if args.mode == 'test':
            tester = Tester_SVL("config.yaml")
            tester.test()

    elif args.learning == 'FSL':
        if args.mode == 'train':
            logger.info('Start few-shot learning')
            trainer = Trainer_FSL("config.yaml")
            trainer.train()

        if args.mode == 'test':
            tester = Tester_FSL("config.yaml")
            tester.test()

    else:
        logger.error('Wrong argument!')