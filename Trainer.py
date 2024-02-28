import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import Variable
from utils.Utils import *
from Loss import Loss, MaskLoss

class Train:
    """
    The Train class sets up train process, combining DataLoader, network, loss, and other modules
    Inputs: config, logger, net, train_data_loader, val_data_loader, device
        - config: config file for init Train class
        - logger: the class<LoggerWriter>, to save model and log while training
        - net: the network to train
        - train_data_loader: the class<DataLoader>, to load train data
        - val_data_loader: the class<DataLoader>, to load valid data
        - device: the device will run
    """
    def __init__(self, config, logger, net, train_data_loader, val_data_loader, device):
        self.net = net
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.logger = logger
        self.device = device

        self.lr = config['train_params']['learning_rate']
        # self.loss = Loss(config['train_params']['loss']).loss()
        self.opt = config['train_params']['optimizer']

        self.epoch = config['train_params']['epoch']
        self.show_steps = config['train_params']['show_steps']
        self.save_mode = config['train_params']['save_mode']

        self.stop_epoch = config['train_params']['early_stop_epoch']
        self.no_improve = 0
        self.stopper = False
        self.best_val_loss = None
        
        self.loss_coef_strategy = config['train_params']['loss_coef']

    def set_opt(self, **kwargs):
        """
        Set optimizer by kwargs. Because the params of each optimizer are quite different, therefore this function set
        optimizer by kwargs<dict>. If parameter is None, set default optimizer via self.lr.
        """
        if kwargs == {}:
            self.opt = eval('optim.'+self.opt)(self.net.parameters(), lr=self.lr)
        else:
            self.opt = eval('optim.'+self.opt)(self.net.parameters(), kwargs)

    def early_stop(self):
        """
        Set early stop strategy for training
        """
        if self.best_val_loss is None:
            self.best_val_loss = np.mean(self.val_loss_dict['pred'])
        else:
            if np.mean(self.val_loss_dict['pred']) < self.best_val_loss:
                self.no_improve = 0
                self.best_val_loss = np.mean(self.val_loss_dict['pred'])
            else:
                self.no_improve += 1
        if self.no_improve == self.stop_epoch:
            self.stopper = True

    def update_learning_rate(self):
        if self.best_val_loss is None:
            self.best_val_loss = np.mean(self.val_loss_dict['pred'])
        else:
            if np.mean(self.val_loss_dict['pred']) < self.best_val_loss:
                self.no_improve = 0
                self.best_val_loss = np.mean(self.val_loss_dict['pred'])
            else:
                self.no_improve += 1
        if self.no_improve == 5:
            lr = self.opt_pred.param_groups[0]['lr']
            for x in self.opt_pred.param_groups:
                x['lr'] = lr * 0.9
            self.no_improve = 0

    def train(self):
        """
        Training process
        """
        iter = 0
        epoch_iter = 0
        epoch_current = 0

        show_loss = []
        train_loss = []

        step_time = time.time()
        # train start
        while epoch_current < self.epoch and (not self.stopper):
            # init optimizer before each batch, set gradients of all model parameters to zero
            self.opt.zero_grad()

            # load data batch
            u_batch, p_batch = self.train_data_loader.load_data()
            u_batch = Variable(torch.FloatTensor(u_batch).to(self.device), requires_grad=False)
            p_batch = Variable(torch.FloatTensor(p_batch).to(self.device), requires_grad=False)

            # predict result
            output, hx = self.net(u_batch)

            # get train loss, and compute the gradient
            loss = self.loss(output, p_batch)
            loss.backward()
            show_loss.append(loss.item())
            train_loss.append(loss.item())

            # update the parameters based on the computed gradients
            self.opt.step()

            # show each train results
            if iter % self.show_steps == 0:
                print('Train Epoch: {} -- Iter: {} -- Loss: {:.4} -- Time: {}'.format(self.train_data_loader.epoch,
                                                                                epoch_iter, np.mean(show_loss),
                                                                                format_runtime(time.time()-step_time)))
                # write train log
                self.logger.write_train_log(epoch_current, epoch_iter, np.mean(show_loss), time.time()-step_time)
                step_time = time.time()
                show_loss = []

            epoch_iter += 1
            # one epoch finish
            if epoch_current != self.train_data_loader.epoch:
                # valid
                self.val_loss, val_time = self.val()
                # save model weight based on val_loss
                self.logger.save_model(self.net, self.val_loss, mode=self.save_mode)
                # write valid log
                self.logger.write_test_log(epoch_current, 0, self.val_loss, val_time)

                epoch_current = self.train_data_loader.epoch
                epoch_iter = 0

                self.early_stop()
                step_time = time.time()
                
    def update_loss_coef(self, epoch_current):
        if epoch_current <= 50:
            coef_1 = 0.1 + (epoch_current/50) * 0.8
            coef_2 = 0.9 - (epoch_current/50) * 0.8
        else:
            coef_1 = 0.9
            coef_2 = 0.1
        
        return coef_1, coef_2
        

                
    def train_mask(self):
        iter = 0
        epoch_iter = 0
        epoch_current = 0
        self.loss = Loss()

        step_time = time.time()

        # init opt
        self.pred_net_params, self.mask_net_params = self.net.get_params()
        # self.opt_pred = eval('optim.' + 'SGD')(self.pred_net_params, lr=self.lr*10, momentum=0.9)
        self.opt_pred = eval('optim.' + 'Adam')(self.pred_net_params, lr=self.lr)
        self.opt_mask = eval('optim.' + self.opt)(self.mask_net_params, lr=self.lr*1e-1)

        # init mask
        batch_size = self.train_data_loader.batch_size
        u_dim = self.train_data_loader.u_dim
        
        # mask_before = torch.empty(int(u_dim / 3)).uniform_(0, 1)
        # mask_before = torch.bernoulli(mask_before)
        mask_before = np.ones(int(u_dim / 3), dtype=float)
        mask_before = torch.FloatTensor(mask_before).to(self.device)
        self.mask_save_no_binary = mask_before.clone()
        mask_save = []
        # init loss
        self.loss4mask = getattr(self.loss, 'MaskLoss')()
        self.loss4pred = getattr(self.loss, 'VarMseLoss')()

        show_loss_dict = {'mask_active': [], 'mask_sim': [], 'mask_total': [],
                          'pred': []}
                          
        self.convert = True

        self.net.mask_net.init_weight()
        self.first_iter = True

        while epoch_current < self.epoch and (not self.stopper):
            # mask_before_ = np.repeat(mask_before, 3)net
            # mask_before_ = np.repeat(np.expand_dims(mask_before, axis=0), batch_size, axis=0)
            mask_before = Variable(mask_before.to(self.device), requires_grad=False)

            # init optimizer before each batch, set gradients of all model parameters to zero
            self.opt_mask.zero_grad()
            u_batch, p_batch = self.train_data_loader.load_data()
            u_batch = Variable(torch.FloatTensor(u_batch).to(self.device), requires_grad=False)
            p_batch = Variable(torch.FloatTensor(p_batch).to(self.device), requires_grad=False)
            mask = self.net.generate_mask(u_batch)

            mask_sigmoid = F.sigmoid(mask)

            loss_activate, loss_sim = self.loss4mask(mask_sigmoid, mask_before)
            
            # if epoch_current < 25:
            #     loss_mask = 0.5 * loss_activate + 0.5 * loss_sim
            # else:
            #     loss_mask = (4e-3*epoch_current + 0.4) + (-4e-3*epoch_current + 0.6) * loss_sim
            # loss_mask = (6e-3*epoch_current + 0.2) * loss_activate + (-6e-3*epoch_current + 0.8) * loss_sim
            # coef_1, coef_2 = self.update_loss_coef(loss_activate, loss_sim)
            '''
            if not self.first_iter:
                loss_mask = 0.5 * loss_activate  + 0.5 * loss_sim
            else:
                loss_mask = loss_activate
                self.first_iter = False
            '''
            if self.loss_coef_strategy == 'dynamic':
                coef_1, coef_2 = self.update_loss_coef(epoch_current)
            else:
                coef_1 = self.loss_coef_strategy
                coef_2 = 1 - self.loss_coef_strategy
            # coef_1, coef_2 = 0.5, 0.5
            loss_mask = coef_1 * loss_activate  + coef_2 * loss_sim
            
            loss_mask.backward()
            show_loss_dict['mask_active'].append(loss_activate.item())
            show_loss_dict['mask_sim'].append(loss_sim.item())
            show_loss_dict['mask_total'].append(loss_mask.item())
            self.opt_mask.step()

            self.opt_pred.zero_grad()
            mask = self.net.generate_mask(u_batch)
            mask_sigmoid = F.sigmoid(mask)
            zero_index = mask_sigmoid.mean(1) == 0
            if zero_index.sum() == batch_size:
                for i in torch.arange(batch_size):
                    mask_sigmoid[i] = mask_before.clone()
            
            
            # mask_binary = self.binarization_mask(mask_sigmoid)
            outputs, hx = self.net.get_prediction(u_batch, mask_sigmoid)
            loss_pred_no_mean = self.loss4pred(outputs, p_batch)
            loss_pred = loss_pred_no_mean.mean()
            loss_pred.backward()
            show_loss_dict['pred'].append(loss_pred.item())

            self.opt_pred.step()

            # if not self.first_iter:
            #     mask_before = self.update_mask(mask_sigmoid, loss_pred_no_mean)
            mask_before = self.update_mask(mask_sigmoid, loss_pred_no_mean, mask_before)
            mask_save.append(self.mask_save_no_binary.cpu().detach().numpy().reshape((11, 11)))

            # show each train results
            if iter % self.show_steps == 0:
                print('Train Epoch: {} -- Iter: {} -- LossMask: {:.4} -- LossMaskActive: {:.4} -- LossMaskSim: {:.4} -- LossPred: {:.4} -- Time: {}'.format(
                    self.train_data_loader.epoch, epoch_iter, np.mean(show_loss_dict['mask_total']), np.mean(show_loss_dict['mask_active']),
                    np.mean(show_loss_dict['mask_sim']), np.mean(np.mean(show_loss_dict['pred'])), format_runtime(time.time() - step_time)))
                # write train log
                self.logger.write_train_log(epoch_current, epoch_iter, np.mean(show_loss_dict['mask_total']), np.mean(show_loss_dict['mask_active']),
                                            np.mean(show_loss_dict['mask_sim']), np.mean(np.mean(show_loss_dict['pred'])), time.time() - step_time)
                step_time = time.time()

                show_loss_dict = {'mask_active': [], 'mask_sim': [], 'mask_total': [],
                                  'pred': []}
            epoch_iter += 1
            # one epoch finish
            if epoch_current != self.train_data_loader.epoch:
                # valid
                self.val_loss_dict, val_time = self.val_mask()
                # save model weight based on val_loss
                if self.loss_coef_strategy == 'dynamic':
                    if epoch_current >= 80:
                        self.logger.save_model(self.net.pred_net, np.mean(self.val_loss_dict['pred']), mode=self.save_mode, prefix='pred')
                        self.logger.save_model(self.net.mask_net, np.mean(self.val_loss_dict['pred']), mode=self.save_mode, prefix='mask')
                else:
                    if epoch_current >= 0:
                        self.logger.save_model(self.net.pred_net, np.mean(self.val_loss_dict['pred']), mode=self.save_mode, prefix='pred')
                        self.logger.save_model(self.net.mask_net, np.mean(self.val_loss_dict['pred']), mode=self.save_mode, prefix='mask')
                # write valid log
                self.logger.write_test_log(epoch_current, 0, np.mean(self.val_loss_dict['mask']), '-', '-', np.mean(self.val_loss_dict['pred']), val_time)
                self.logger.mask_save(epoch_current, mask_save)
                mask_save = []

                epoch_current = self.train_data_loader.epoch
                epoch_iter = 0

                # self.update_learning_rate()
                step_time = time.time()
         
    def sum_mask_point(self, mask_binary):
        return torch.sum(mask_binary, dim=1)

    def binarization_mask(self, mask_sigmoid):
        mask_binary = mask_sigmoid.clone()
        mask_binary[mask_binary >= 0.5] = 1
        mask_binary[mask_binary < 0.5] = 0
        return mask_binary

    def update_mask(self, mask, loss, mask_before):
        
        mask_ = mask.clone()
        # loss_ = loss.clone()
        # loss_ = loss_.reshape((loss_.shape[0], int(loss_.shape[1]*loss_.shape[2])))
        # loss_mean_ = torch.mean(loss_, dim=1)
        loss_mean_ = loss.clone()
        if self.convert:
            min_index = torch.argmin(loss_mean_)
            mask_return = mask_[min_index]
        else:
            max_index = torch.argmax(loss_mean_)
            mask_return = mask_[max_index]
        
        min_index = torch.argmin(loss_mean_)
        mask_return = mask_[min_index] 
        self.mask_save_no_binary = mask_return.clone()   
        
        self.convert = not self.convert
        # min_index = torch.argmin(loss_mean_)
        # mask_return = mask_[min_index]
        mask_return[mask_return >= 0.5] = 1
        mask_return[mask_return < 0.5] = 0
        if not torch.any(mask_return.bool()):
            i_ = torch.randint(0, 100, (1, ))
            # mask_return[i_] = 1.0
            mask_return = mask_before
            self.mask_save_no_binary = mask_return.clone()
        
        # mask_ = mask.clone()
        # mask_binary = self.binarization_mask(mask_)
        # mask_return = torch.all(mask_binary.bool(), 0).float()
        # return mask_return
        '''
        mask_ = mask.clone()
        mask_mean = torch.mean(mask_, dim=0)
        mask_binary = self.binarization_mask(mask_mean)
        '''
        return mask_return

    def val(self):
        """
        Valid process
        """
        self.val_data_loader.reset()
        val_loss = []
        val_iter = 0
        # set network as evaluation mode (disable BN layers and dropout)
        self.net.eval()
        step_time = time.time()
        while self.val_data_loader.epoch < 1:
            # load data batch
            u_batch, p_batch = self.val_data_loader.load_data()
            u_batch = Variable(torch.FloatTensor(u_batch).to(self.device), requires_grad=False)
            p_batch = Variable(torch.FloatTensor(p_batch).to(self.device), requires_grad=False)
            # predict results
            output, hx = self.net(u_batch)
            # get valid loss
            loss = self.loss(p_batch, output)
            val_loss.append(loss.item())
            val_iter += 1
        val_time = time.time() - step_time
        print('Valid Epoch: {} ------------- Loss: {:.4} -- Time: {}'.format(self.train_data_loader.epoch, np.mean(val_loss),
                                                                             format_runtime(val_time)))
        # reset network as train mode
        self.net.train()
        return np.mean(val_loss), val_time

    def val_mask(self):
        """
        Valid process
        """
        self.val_data_loader.reset()
        val_loss = []
        val_loss_dict = {'mask': [], 'pred': []}
        val_iter = 0
        # set network as evaluation mode (disable BN layers and dropout)
        self.net.eval()
        step_time = time.time()
        activate_points = []
        while self.val_data_loader.epoch < 1:
            # load data batch
            u_batch, p_batch = self.val_data_loader.load_data()
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
        print('Valid Epoch: {} -- LossMask: {:.4} -- ActiveMask: {:.4}-- LossPred: {:.4} -- Time: {}'.format(
            self.train_data_loader.epoch,
            np.mean(val_loss_dict['mask']),
            np.mean(activate_points),
            np.mean(val_loss_dict['pred']),
            format_runtime(val_time)))
        # reset network as train mode
        self.net.train()
        return val_loss_dict, val_time


