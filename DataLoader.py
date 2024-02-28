import torch
import os
import numpy as np

class DataLoader:
    def __init__(self, config, mode):
        """
        The DataLoader loads data as a batch data
        Inputs: config, mode
            - config: config file for init DataLoader class
            - mode: the mode of Dataloader running, limited in ['train', 'test', 'predict']
        Extrnal functions: reset(), load_data()
            - reset(): using this function will reset the DataLoader, that is the load pointer will be set as 0
            - load_data(): this function loads data as one batch and follows the load mode. When mode is
                ['train', 'test'] both u_data and p_data will be loaded. When mode is 'predict', only p_data will be
                loaded.
        Example:
            >>> data_index_file_path = './test_data/data_index.txt'
            >>> config = {'dataset_index': data_index_file_path, 'steady_time': 0, 'period_time': 616, 'n_frame': 0,
            >>>     'batch_size': 5, 'epoch': 10, 'random_period': True}
            >>> train_loader = DataLoader(config, 'test')
            >>> while train_loader.epoch < 1:
            >>>     batch_u, batch_p = train_loader.load_data()
            >>>     print(batch_u.shape)
        """
        self.dataset_index_file = config['{}_dataset_index'.format(mode)]
        self.steady_threshold = config['steady_time']
        self.period_time = config['period_time']
        
        self.batch_size = config['batch_size']
        self.random_period_flag = config['random_period']
        self.mode = mode

        if 'augmentation' in config:
            self.augmentation_params = config['augmentation']
        else:
            self.augmentation_params = None
        
        self.augmentation()

        if mode not in ['train', 'test', 'predict']:
            raise ValueError('The mode of DataLoader must be limited in ["train", "test", "predict"], '
                             'error {}'.format(self.mode))
        if mode in ['train', 'test']:
            self.load_data = self.load_data_train_test
        else:
            self.load_data = self.load_data_predict

        self.read_file_list()
        self.read_data()

        self.file_pointer = 0
        self.data_pointer = 0
        self.epoch = 0

        self.generate_data_index()
        
        if self.mode == 'train':
            self.random_file()

            
    def random_file(self):
        seed_ = np.arange(len(self.u_data), dtype=int)
        np.random.shuffle(seed_)
        self.u_data = self.u_data[seed_]
        self.p_data = self.p_data[seed_]
        self.file_pointer = 0

    def random_period(self):
        np.random.shuffle(self.data_index)

    def generate_data_index(self):
        self.data_size = self.u_data[self.file_pointer].shape[0]
        if self.mode == 'train' and self.augmentation_params is not None:
            if 'repeat' in self.augment_modes:
                self.data_index = np.arange(self.steady_threshold, self.data_size, int(self.period_time/(self.augment_repeat+1)), dtype=int)
                self.data_index = self.data_index[:-self.augment_repeat-1]
            else:
                self.data_index = np.arange(self.steady_threshold, self.data_size, self.period_time, dtype=int)
        else:
            self.data_index = np.arange(self.steady_threshold, self.data_size, self.period_time, dtype=int)
        # print(self.data_index)


        self.data_period_num = len(self.data_index)
        if self.mode == 'train' and self.random_period_flag == True:
            self.random_period()


    def reset(self):
        self.file_pointer = 0
        self.data_pointer = 0
        self.epoch = 0

    def get_file_name(self):
        if self.mode in ['train', 'test']:
            if self.file_pointer >= self.file_num:
                return 'end', 'end'
            return os.path.basename(self.u_data_file[self.file_pointer]), \
                   os.path.basename(self.p_data_file[self.file_pointer])
        else:
            if self.file_pointer >= self.file_num:
                return 'end'
            return os.path.basename(self.u_data_file[self.file_pointer])

    def read_file_list(self):
        """
        According to the index file of data set, read file list
        """
        if self.mode in ['train', 'test']:
            self.u_data_file, self.p_data_file = [], []
            for line in open(self.dataset_index_file, 'r'):
                u, p = line.replace('\r\n', '').replace('\n', '').split(',')
                self.u_data_file.append(u)
                self.p_data_file.append(p)
            self.file_num = len(self.u_data_file)
        elif self.mode == 'predict':
            self.u_data_file = []
            for line in open(self.dataset_index_file, 'r'):
                u = line.replace('\r\n', '').replace('\n', '')
                self.u_data_file.append(u)
            self.file_num = len(self.u_data_file)



    def read_data(self):
        """
        This function will read two type data into memory and their format are shown as follows:
            u_data<np.array> structure: (file_no, np.array(time_step, u_dim)) (NOTE: time_step is determined by each file)
            p_data<np.array> structure: (file_no, np.array(time_step, p_dim)) (NOTE: time_step is determined by each file)
        And this function will be based on self.mode to read.
        Moreover, the function will check the dimension of each data to ensure that same data has the same dimension
        """
        # Read data into memory
        if self.mode == 'predict':
            self.u_data = []
            for u in self.u_data_file:
                u_data = np.loadtxt(u, delimiter=',', dtype=np.float32)
                self.u_data.append(u_data)
        else:
            self.u_data, self.p_data = [], []
            for u, p in zip(self.u_data_file, self.p_data_file):
                u_data = np.loadtxt(u, delimiter=',', dtype=np.float32)
                p_data = np.loadtxt(p, delimiter=',', dtype=np.float32)

                if u_data.shape[0] != p_data.shape[0]:
                    raise ValueError("Error matching of the number of velocity and pressure data, {} != {}, in file {} and {}".
                           format(u_data.shape[0], p_data.shape[0], u, p))
                else:
                    self.u_data.append(u_data)
                    self.p_data.append(p_data)

        # Check dimension

        u_dim_check = [x.shape[1] for x in self.u_data]
        if len(set(u_dim_check)) != 1:
            raise ValueError("The number of point in all velocity must be same. Velocity point has {} formats: {}".
                   format(len(set(u_dim_check)), set(u_dim_check)))
        else:
            self.u_dim = self.u_data[0].shape[1]
            self.u_dtype = self.u_data[0].dtype
        del u_dim_check

        self.u_data = np.array(self.u_data)

        if self.mode in ['train', 'test']:
            p_dim_check = [x.shape[1] for x in self.p_data]
            if len(set(p_dim_check)) != 1:
                raise ValueError("The number of point in all pressure must be same. Velocity point has {} formats: {}".
                       format(len(set(p_dim_check)), set(p_dim_check)))
            else:
                self.p_dim = self.p_data[0].shape[1]
                self.p_dtype = self.p_data[0].dtype
            del p_dim_check

            self.p_data = np.array(self.p_data)


    def load_data_train_test(self):
        """
        The function load u_data and p_data as one batch. Meanwhile, if one epoch finish, the function will random the
        whole dataset.
        :return:
            batch_u<np.array> structure: (batch_size, time_step, u_dim)
            batch_p<np.array> structure: (batch_size, time_step, p_dim)
        """
        batch_u = np.zeros((self.batch_size * self.period_time,) + (self.u_dim,), dtype=self.u_dtype)
        batch_p = np.zeros((self.batch_size * self.period_time,) + (self.p_dim,), dtype=self.p_dtype)
        read_ = self.batch_size
        while read_ > 0:
            next_read = self.data_period_num - self.data_pointer
            start_pointer = self.batch_size - read_
            # The rest data in one file is not enough for one batch
            if read_ - next_read >= 0:
                # generate index following period_time
                # example period_time = 5, and next_read = 2, because of random load, therefore the batch_index maybe
                # [15, 16, 17, 18, 19, 0, 1, 2, 3, 4]
                batch_index = np.tile(np.arange(self.period_time, dtype=int), next_read) + \
                              np.repeat(self.data_index[self.data_pointer:], self.period_time)

                end_pointer = start_pointer + next_read

                start_pointer = start_pointer * self.period_time
                end_pointer = end_pointer * self.period_time

                # load data by batch_index
                batch_u[start_pointer: end_pointer, :] = self.u_data[self.file_pointer][batch_index, :]
                batch_p[start_pointer: end_pointer, :] = self.p_data[self.file_pointer][batch_index, :]

                read_ = read_ - next_read

                if self.file_pointer + 1 == self.file_num:
                    # One epoch finishes
                    self.random_file()
                    self.epoch = self.epoch + 1
                else:
                    # One data_file finishes
                    self.file_pointer += 1
                # read and generate new data index
                self.generate_data_index()
                self.data_pointer = 0
            # The rest data in one file is enough for one batch
            else:
                batch_index = np.tile(np.arange(self.period_time, dtype=int), read_) + \
                              np.repeat(self.data_index[self.data_pointer: self.data_pointer+read_], self.period_time)

                start_pointer = start_pointer * self.period_time
                batch_u[start_pointer:, :] = self.u_data[self.file_pointer][batch_index, :]
                batch_p[start_pointer:, :] = self.p_data[self.file_pointer][batch_index, :]

                self.data_pointer = self.data_pointer + read_

                read_ = 0
        batch_u = batch_u.reshape((self.batch_size, self.period_time, self.u_dim))
        batch_p = batch_p.reshape((self.batch_size, self.period_time, self.p_dim))

        # return self.norm_data(batch_u), self.norm_data(batch_p)
        return batch_u, batch_p

    def load_data_predict(self):
        """
        This function is same as load_data_train_test()
        """
        batch_u = np.zeros((self.batch_size * self.period_time,) + (self.u_dim,), dtype=self.u_dtype)
        read_ = self.batch_size
        while read_ > 0:
            next_read = self.data_period_num - self.data_pointer
            start_pointer = self.batch_size - read_
            if read_ - next_read >= 0:
                batch_index = np.tile(np.arange(self.period_time, dtype=int), next_read) + \
                              np.repeat(self.data_index[self.data_pointer:], self.period_time)

                end_pointer = start_pointer + next_read

                start_pointer = start_pointer * self.period_time
                end_pointer = end_pointer * self.period_time

                batch_u[start_pointer: end_pointer, :] = self.u_data[self.file_pointer][batch_index, :]

                read_ = read_ - next_read

                if self.file_pointer + 1 == self.file_num:
                    self.epoch = self.epoch + 1
                else:
                    self.generate_data_index()
                    self.data_pointer = 0

                self.file_pointer += 1


                batch_u = batch_u.reshape((self.batch_size-read_, self.period_time, self.u_dim))
                return batch_u

            else:
                batch_index = np.tile(np.arange(self.period_time, dtype=int), read_) + \
                              np.repeat(self.data_index[self.data_pointer: self.data_pointer+read_], self.period_time)

                start_pointer = start_pointer * self.period_time
                batch_u[start_pointer:, :] = self.u_data[self.file_pointer][batch_index, :]

                self.data_pointer = self.data_pointer + read_

                read_ = 0
        batch_u = batch_u.reshape((self.batch_size, self.period_time, self.u_dim))

        return batch_u
 
    def norm_data(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def augmentation(self):
        if self.augmentation_params is not None:
            self.augment_modes = self.augmentation_params['mode']
            if 'repeat' in self.augment_modes:
                self.augment_repeat = self.augmentation_params['repeat_period']

        
        
