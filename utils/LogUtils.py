import os
import time
import json
from shutil import copyfile, rmtree

import numpy as np

from network import *

class LoggerWriter:
    """
    LoggerWriter completes the functions implementation of log writing and model saving
    Inputs: config, checkpoint
        - config: the global config file for whole application
        - checkpoint: the checkpoint path to load, default is None
    """
    def __init__(self, config, checkpoint=None):
        self.config = config
        self.checkpoint = checkpoint
        self.model_save_index = 0
        self.last_metric = {}

        self.net_name = self.config['network_params']['name']
        self.lr_name = self.config['train_params']['learning_rate']
        self.loss_name = self.config['train_params']['loss']

        self.proj_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        self.init_path()
        self.set_log_format()


    def init_path(self):
        """
        init path based on checkpoint path, if it is None, init path based on time, network's name, loss's name, and lr
        """
        if self.checkpoint is None:
            log_root = self.config['log_params']['log_root']
            if not os.path.exists(log_root):
                raise RuntimeError('Log root directory "{}" does not exist'.format(log_root))
            create_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
            
            if 'comparison_exp' in self.config['log_params']:
                if self.config['log_params']['comparison_exp'] == True:
                    S = self.config['train_params']['loss_coef']
                    R = '-'.join([str(i) for i in self.config['network_params']['hidden_size']])
                    U = '-'.join([str(i) for i in self.config['network_params']['mapping_net']])
                    
                    self.log_dir = os.path.join(self.config['log_params']['log_root'], '{}_S_{}_R_{}_U_{}'.format(
                        create_time, S, R, U))
                else:
                    self.log_dir = os.path.join(self.config['log_params']['log_root'], '{}_{}_{}_{}'.format(
                        create_time, self.net_name, self.lr_name, self.loss_name))
            else:
                self.log_dir = os.path.join(self.config['log_params']['log_root'], '{}_{}_{}_{}'.format(
                    create_time, self.net_name, self.lr_name, self.loss_name))

            # self.log_dir = os.path.join(self.config['log_params']['log_root'], '{}_{}_{}_{}'.format(
            #     create_time, self.net_name, self.lr_name, self.loss_name))

            os.mkdir(self.log_dir)

            self.config_save_path = os.path.join(self.log_dir, 'config')
            self.weight_save_path = os.path.join(self.log_dir, 'weight')
            self.model_save_path = os.path.join(self.log_dir, 'model')
            self.loss_save_path = os.path.join(self.log_dir, 'loss')


            os.mkdir(self.config_save_path)
            os.mkdir(self.weight_save_path)
            os.mkdir(self.model_save_path)
            os.mkdir(self.loss_save_path)

            if 'Mask' in self.net_name:
                self.mask_save_path = os.path.join(self.log_dir, 'mask')
                os.mkdir(self.mask_save_path)

            save_config_file = open(os.path.join(self.config_save_path, 'config.json'), 'w')
            json.dump(self.config, save_config_file, indent=4)
            save_config_file.close()

            copyfile(os.path.join(self.proj_root, 'network.py'), os.path.join(self.model_save_path, 'network.py'))
            copyfile(os.path.join(self.proj_root, 'Loss.py'), os.path.join(self.model_save_path, 'Loss.py'))
            copyfile(os.path.join(self.proj_root, 'Trainer.py'), os.path.join(self.model_save_path, 'Trainer.py'))

        else:
            if not os.path.exists(self.checkpoint):
                raise RuntimeError('Checkpoint directory "{}" does not exist'.format(self.checkpoint))
            self.log_dir = self.checkpoint

            self.config_save_path = os.path.join(self.log_dir, 'config')
            self.weight_save_path = os.path.join(self.log_dir, 'weight')
            self.model_save_path = os.path.join(self.log_dir, 'model')
            self.loss_save_path = os.path.join(self.log_dir, 'loss')

            if 'Mask' in self.net_name:
                self.mask_save_path = os.path.join(self.log_dir, 'mask')

    def set_log_format(self, log_header=None):
        """
        This function sets the table header of log file, if log_header is None, set as default format
        """
        if log_header is None:
            self.log_header = 'Epoch,Iter,Loss-{},Time\n'.format(self.loss_name)
            self.log_format = '{},{},{},{}\n'
        else:
            self.log_header = log_header
            self.log_format = ','.join(['{}']*len(self.log_header.split(',')))+'\n'

    def init_logs(self):
        """
        Create log file
        """
        self.train_log = os.path.join(self.loss_save_path, 'train_loss.csv')
        self.test_log = os.path.join(self.loss_save_path, 'test_loss.csv')
        if not os.path.exists(self.train_log):
            with open(self.train_log, 'w') as f:
                f.write(self.log_header)
                f.close()
        if not os.path.exists(self.test_log):
            with open(self.test_log, 'w') as f:
                f.write(self.log_header)
                f.close()

    def write_train_log(self, *args):
        with open(self.train_log, 'a') as f:
            f.write(self.log_format.format(*args))
            f.close()
    def write_test_log(self, *args):
        with open(self.test_log, 'a') as f:
            f.write(self.log_format.format(*args))
            f.close()

    def load_model(self, model_name=None, device='cuda'):
        """
        Load saved model based on the network and weight in checkpoint path
        """
        net = eval(self.net_name)(self.config['network_params'], device)  # load model based on network name in config
        if 'Mask' in self.net_name:
            if model_name is not None:
                pred_model_path = os.path.join(self.weight_save_path, 'pred_' + model_name + '.pkl')
                mask_model_path = os.path.join(self.weight_save_path, 'mask_' + model_name + '.pkl')
                self.model_name = model_name
            elif os.path.exists(os.path.join(self.weight_save_path, 'best_pred_model.pkl')) \
                and os.path.exists(os.path.join(self.weight_save_path, 'best_mask_model.pkl')):
                pred_model_path = os.path.join(self.weight_save_path, 'best_pred_model.pkl')
                mask_model_path = os.path.join(self.weight_save_path, 'best_mask_model.pkl')
                self.model_name = 'best'
            else:
                raise RuntimeError('The model "{}" dose not exist'.format(model_name))
            net.pred_net.load_state_dict(torch.load(pred_model_path, map_location=torch.device(device)))
            net.mask_net.load_state_dict(torch.load(mask_model_path, map_location=torch.device(device)))
        else:
            if model_name is not None:
                model_path = os.path.join(self.weight_save_path, model_name + '.pkl')
                self.model_name = model_name
            elif os.path.exists(os.path.join(self.weight_save_path, 'best_model.pkl')):
                # if model_name is None, load best_model.pkl as default weight
                model_path = os.path.join(self.weight_save_path, 'best_model.pkl')
                self.model_name = 'best'
            else:
                raise RuntimeError('The model "{}" dose not exist'.format(model_name))

            net.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        return net

    def save_model(self, net, metric, mode='min', prefix=None):
        """
        Save the weight of model
        Paramters:
            - net: network<torch.nn.Module>
            - metric: the evaluation metrics which the model saving is based on
            - mode: mode limited in ['min', 'max'], if mode is 'min', select the minimal metrics as best model
        """
        if prefix is None:
            model_name = 'model'
        else:
            model_name = prefix + '_model'
        # torch.save(net.state_dict(), os.path.join(self.weight_save_path, '{}_{}.pkl'.format(model_name, self.model_save_index)))
        self.model_save_index += 1
        if prefix not in self.last_metric:
            # torch.save(net.state_dict(), os.path.join(self.weight_save_path, 'best_{}.pkl'.format(model_name)))
            self.last_metric[prefix] = metric
        else:
            if mode == 'min':
                if metric < self.last_metric[prefix]:
                    torch.save(net.state_dict(), os.path.join(self.weight_save_path, 'best_{}.pkl'.format(model_name)))
                    self.last_metric[prefix] = metric
            elif mode == 'max':
                if metric > self.last_metric[prefix]:
                    torch.save(net.state_dict(), os.path.join(self.weight_save_path, 'best_{}.pkl'.format(model_name)))
                    self.last_metric[prefix] = metric
            else:
                raise ValueError('Save mode must be in ["max", "min"], error {}'.format(mode))

    def set_predict_save(self):
        self.predict_save_root = os.path.join(self.log_dir, 'predict_save')
        if os.path.exists(self.predict_save_root):
            return
            # rmtree(self.predict_save_root)
        os.mkdir(self.predict_save_root)

    def write_predict(self, file_name, prediction):
        save_file_path = os.path.join(self.predict_save_root, file_name)
        if not os.path.exists(save_file_path):
            save_file = open(save_file_path, 'w')
            save_file.close()
        prediction_ = prediction.cpu().detach().numpy()
        prediction_ = prediction_.reshape(prediction_.shape[0]*prediction_.shape[1], prediction_.shape[2])
        with open(save_file_path, 'a') as save_file:
            np.savetxt(save_file, prediction_, delimiter=',')
            save_file.close()

    def mask_save(self, epoch, mask):
        mask = np.array(mask)
        np.save(os.path.join(self.mask_save_path, 'mask{}.npy'.format(epoch)), mask)




