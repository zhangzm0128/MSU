import os
import json
import argparse
import time
from network import *
from DataLoader import DataLoader
from utils.LogUtils import LoggerWriter

from Trainer import Train
from test import Test
from predictor import Predict

# Load external parameters
parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str, default='config.json',
                    help='the path of global config file.')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='the path of checkpoint and program will run checkpoint data.')
parser.add_argument('--model_name', type=str, default=None)
parser.add_argument('--mode', type=str, default='train',
                    help='the mode of app will run, plz choose among ["train", "test", "predict"]')
parser.add_argument('--device', type=str, default='cpu',
                    help='the device of app will run, plz choose among ["cuda", "cpu"]')

args = parser.parse_args()

config_file = open(args.config, 'r').read()
config = json.loads(config_file)

model_name = args.model_name
checkpoint = args.checkpoint
mode = args.mode
device = args.device

# Select mode to run
if mode == 'train':
    net_name = config['network_params']['name']
    net = eval(net_name)(config['network_params'], device)
    train_data_loader = DataLoader(config['data_params'], 'train')
    val_data_loader = DataLoader(config['data_params'], 'test')

    logger = LoggerWriter(config, checkpoint)
    logger.set_log_format('Epoch,Iter,Loss-Mask,Loss-MaskActive,Loss-MaskSim,Loss-MSE,Time\n')
    logger.init_logs()
    trainer = Train(config, logger, net, train_data_loader, val_data_loader, device)
    # trainer.set_opt()
    trainer.train_mask()
elif mode == 'test':
    logger = LoggerWriter(config, checkpoint)
    net = logger.load_model(device=device)
    test_data_loader = DataLoader(config['data_params'], 'test')
    tester = Test(config, logger, net, test_data_loader, device)
    tester.test_mask()
elif mode == 'predict':
    config['data_params']['batch_size'] = 1
    logger = LoggerWriter(config, checkpoint)
    net = logger.load_model(device=device, model_name=model_name)
    pre_data_loader = DataLoader(config['data_params'], 'predict')
    predictor = Predict(config, logger, net, pre_data_loader, device)
    predictor.predict_mask()
else:
    print('Plz choose correct mode among ["train", "test", "predict"]')
