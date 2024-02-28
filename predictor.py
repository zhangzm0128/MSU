import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
from utils.Utils import *
from Loss import Loss
from DataLoader import DataLoader
from utils.LogUtils import LoggerWriter
from utils.Utils import binarization_mask, sum_mask_point

class Predict:
    def __init__(self, config, logger, net, pre_data_loader, device):
        self.config = config
        self.net = net
        self.pre_data_loader = pre_data_loader
        self.device = device
        self.logger = logger
        self.net.eval()

        self.logger.set_predict_save()

    def predict(self):
        file_name = self.pre_data_loader.get_file_name()

        step_time = time.time()
        while self.pre_data_loader.epoch < 1:
            u_batch = self.pre_data_loader.load_data()
            u_batch = Variable(torch.FloatTensor(u_batch).to(self.device), requires_grad=False)
            output, hx = self.net(u_batch)

            if file_name != self.pre_data_loader.get_file_name():
                print('Predict {} -- Time: {}'.format(file_name, format_runtime(time.time()-step_time)))
                self.logger.write_predict(file_name, output)
                file_name = self.pre_data_loader.get_file_name()
            else:
                self.logger.write_predict(file_name, output)

    def predict_mask(self):
        file_name = self.pre_data_loader.get_file_name()
        model_name = self.logger.model_name

        step_time = time.time()
        self.net.eval()
        activate_points = []
        mask_save = []
        hx = None

        while self.pre_data_loader.epoch < 1:
            u_batch = self.pre_data_loader.load_data()
            u_batch = Variable(torch.FloatTensor(u_batch).to(self.device), requires_grad=False)

            # predict mask
            mask = self.net.generate_mask(u_batch)
            mask_sigmoid = F.sigmoid(mask)
            # predict results
            mask_binary = binarization_mask(mask_sigmoid)
            activate_points.append(sum_mask_point(mask_binary).cpu().detach().numpy())
            output, hx = self.net.get_prediction(u_batch, mask_binary, hx)

            mask_save += mask_binary.cpu().detach().numpy().tolist()

            if file_name != self.pre_data_loader.get_file_name():
                print('Predict {} -- Time: {}'.format(file_name, format_runtime(time.time()-step_time)))
                self.logger.write_predict('{}_{}'.format(model_name, file_name), output)
                self.logger.mask_save(model_name, mask_save)
                file_name = self.pre_data_loader.get_file_name()
            else:
                self.logger.write_predict('{}_{}'.format(model_name, file_name), output)
