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

class Test:
    def __init__(self, config, logger, net, test_data_loader, device):
        self.config = config
        self.net = net
        self.test_data_loader = test_data_loader
        self.device = device
        self.logger = logger
        self.loss = Loss()
        self.loss4mask = getattr(self.loss, 'MaskLoss')()
        self.loss4pred = getattr(self.loss, 'MSE')()

        self.net.eval()


    def binarization_mask(self, mask_sigmoid):
        mask_binary = mask_sigmoid.clone()
        mask_binary[mask_binary > 0.5] = 1
        mask_binary[mask_binary <= 0.5] = 0
        return mask_binary

    def sum_mask_point(self, mask_binary):
        return torch.sum(mask_binary, dim=1)

    def test(self):
        self.test_data_loader.reset()
        val_loss = []
        val_iter = 0
        # set network as evaluation mode (disable BN layers and dropout)
        self.net.eval()
        step_time = time.time()
        while self.test_data_loader.epoch < 1:
            # load data batch
            u_batch, p_batch = self.test_data_loader.load_data()
            u_batch = Variable(torch.FloatTensor(u_batch).to(self.device), requires_grad=False)
            p_batch = Variable(torch.FloatTensor(p_batch).to(self.device), requires_grad=False)
            # predict results
            output, hx = self.net(u_batch)
            # get valid loss
            loss = self.loss4pred(p_batch, output)
            val_loss.append(loss.mean().item())
            val_iter += 1
        val_time = time.time() - step_time
        print('Test ------------- Loss: {:.4} -- Time: {}'.format(np.mean(val_loss), format_runtime(val_time)))
        # reset network as train mode
        self.net.train()
        return np.mean(val_loss), val_time

    def test_mask(self):
        self.test_data_loader.reset()
        val_loss = []
        val_loss_dict = {'mask': [], 'pred': []}
        val_iter = 0
        # set network as evaluation mode (disable BN layers and dropout)
        self.net.eval()
        step_time = time.time()
        activate_points = []
        while self.test_data_loader.epoch < 1:
            # load data batch
            u_batch, p_batch = self.test_data_loader.load_data()
            u_batch = Variable(torch.FloatTensor(u_batch).to(self.device), requires_grad=False)
            p_batch = Variable(torch.FloatTensor(p_batch).to(self.device), requires_grad=False)
            # predict mask
            mask = self.net.generate_mask(u_batch)
            mask_sigmoid = F.sigmoid(mask)
            loss_activate = self.loss4mask(mask_sigmoid)
            val_loss_dict['mask'].append(loss_activate.item())
            # predict results
            mask_binary = self.binarization_mask(mask_sigmoid)
            activate_points.append(self.sum_mask_point(mask_binary).cpu().detach().numpy())
            output, hx = self.net.get_prediction(u_batch, mask_binary)
            # get valid loss
            loss_pred = self.loss4pred(p_batch, output)
            val_loss_dict['pred'].append(loss_pred.mean().item())
            val_iter += 1
        val_time = time.time() - step_time
        print('Test: LossMask: {:.4} -- ActiveMask: {:.4}-- LossPred: {:.4} -- Time: {}'.format(np.mean(val_loss_dict['mask']),
                                                                                                np.mean(activate_points),
                                                                                                np.mean(val_loss_dict['pred']),
                                                                                                format_runtime(val_time)))
                                                                                                
                                                                                                
