from audioop import bias
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RFP(nn.Module):  # RNN based Fluid Prediction
    def __init__(self, config, device='cuda'):
        """
        This network is based on RNN cell for fluid prediction
        Inputs: config, device
            - config: config file for init RFP class
            - device: the device will run
        Output: outputs, hx
            - outputs of shape (batch_size, seq_len, output_dim): tensor represents the prediction results, where
                seq_len is the length of sequence for RNN processing, output_dim is dimension of output data
            - hx of shape (batch_size, last hidden layer size): tensor represents the last hidden layers
        Example:
            >>> config = {'input_size': 60, 'hidden_size': [128, 256, 128], 'output_size': 100,
            >>>  'seq_len': 50, 'cell_type': 'GRUCell', 'num_cells': 3, 'same_linear': False, 'device': 'cuda'}
            >>> net = RFP(config)
            >>> test_input = torch.randn(128, 50, 60).to('cuda')
            >>> output, hx = net(test_input)
        """
        super(RFP, self).__init__()
        # init parameters
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.output_size = config['output_size']
        self.seq_len = config['seq_len']

        self.cell_type = config['cell_type']
        self.num_cells = config['num_cells']
        self.same_linear = config['same_linear']

        self.device = device

        # select cell type
        self.cells = []
        if self.cell_type == 'RNNCell':
            cell = nn.RNNCell
        elif self.cell_type == 'LSTMCell':
            cell = nn.LSTMCell
        elif self.cell_type == 'GRUCell':
            cell = nn.GRUCell
        else:
            raise ValueError("The type of RNN cell must be in ['RNNCell', 'LSTMCell', 'GRUCell'], error {}".
                   format(self.cell_type))

        # init hidden size
        if not isinstance(self.hidden_size, list) and self.num_cells == 1:
            self.hidden_size = [self.hidden_size] * self.num_cells
        elif isinstance(self.hidden_size, list):
            if len(self.hidden_size) != self.num_cells:
                raise ValueError("The length of hidden_size list must be equal as num_cells, error {} != {}".
                       format(len(self.hidden_size), self.num_cells))

        # init linear layers
        if self.same_linear == True:
            linear = nn.Linear(self.hidden_size[-1], self.output_size).to(self.device)
            self.linears = [linear] * self.seq_len
        else:
            self.linears = []
            for x in range(self.seq_len):
                self.linears.append(nn.Linear(self.hidden_size[-1], self.output_size).to(self.device))

        # init each cell
        for h_size in self.hidden_size:
            if self.cells == []:
                self.cells.append(cell(self.input_size, h_size).to(self.device))
            else:
                self.cells.append(cell(last_h_size, h_size).to(self.device))
            last_h_size = h_size

        self.cells = nn.ModuleList(self.cells) # register
        self.linears = nn.ModuleList(self.linears)

    def set_zero_state(self, input):
        # set default zero state with shape (batch_size, first_hidden_size)
        zero_state = []
        for h_size in self.hidden_size:
            zero_state.append(torch.zeros(input.size(0), h_size, dtype=input.dtype, device=input.device).to(self.device))
        return zero_state

    def forward(self, inputs, zero_state=None):
        # inputs shape [batch_size, seq_len, dim]
        # if zero_state is None, the zero_state follows self.set_zero_state()
        if zero_state is None:
            hx = self.set_zero_state(inputs)
        else:
            hx = zero_state
        outputs = []

        for x in range(self.seq_len):
            for cell_no, cell in enumerate(self.cells):
                if cell_no == 0:
                    hx[cell_no] = cell(inputs[:, x, :], hx[cell_no])
                else:
                    hx[cell_no] = cell(hx[cell_no-1], hx[cell_no])
            outputs.append(self.linears[x](hx[-1]))
            # outputs.append(hx[-1])

        outputs = torch.stack(outputs, dim=1)
        return outputs, hx

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Unet(nn.Module):
    def __init__(self, n_channels, n_classes, config, device='cuda'):
        super().__init__()
        self.d = int(np.sqrt(config['input_size'] / 3))
        self.q = config['seq_len']
        self.i = config['input_size']
        self.device = device
        self.first_channel = config['mapping_net'][0]
        self.second_channel = config['mapping_net'][1]
        # input shape (batch_size, seq_len, height * weight * 3)

        if self.d % 2 == 0:
            self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.down_conv_1 = DoubleConv(n_channels, self.first_channel).to(self.device)
        self.down_conv_2 = DoubleConv(self.first_channel, self.second_channel).to(self.device)

        self.up_deconv_2 = nn.ConvTranspose2d(self.second_channel, self.first_channel, kernel_size=2, stride=2).to(self.device)
        self.up_conv_2 = DoubleConv(self.second_channel, self.first_channel).to(self.device)

        # self.up_deconv_1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2).to(self.device)
        
        self.conv_1x1 = nn.Conv2d(self.first_channel, n_classes, kernel_size=1, stride=1).to(self.device)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def pre_process(self, x):
        # x shape (batch, seq_len, height * weight * 3)
        # height * weight * 3 = [point_0x, point_0y, point_0z, ..., point_nx, point_ny, point_nz]
        # x_reshape shape (batch, seq_len, 3, height, weight)
        b = x.shape[0]
        x_reshape = torch.zeros((b, self.q, 3, self.d, self.d))
        x_reshape[:, :, 0, :, :] = x[:, :, 0:self.i:3].reshape((b, self.q, self.d, self.d))
        x_reshape[:, :, 1, :, :] = x[:, :, 1:self.i + 1:3].reshape((b, self.q, self.d, self.d))
        x_reshape[:, :, 2, :, :] = x[:, :, 2:self.i + 2:3].reshape((b, self.q, self.d, self.d))
        return x_reshape

    def forward(self, x):
        x_reshape = self.pre_process(x)
        x_mean = torch.mean(x_reshape, dim=1).to(self.device)

        x1 = self.down_conv_1(x_mean)
        x2 = self.max_pool(x1)
        x2 = self.down_conv_2(x2)

        if self.d % 2 != 0:
            d2 = self.up_deconv_2(x2)[:, :, :-1, :-1]
        # d2 = F.pad(self.up_deconv_2(x2), [-1, 0, -1, 0])
        
        d2 = torch.cat([x1, d2], dim=1)
        d2 = self.up_conv_2(d2)

        d1 = self.conv_1x1(d2)
        out = torch.squeeze(d1, dim=1).view(d1.size(0), -1)  # shape (batch_size, height, weight)

        return out

class MSU():
    def __init__(self, config, device='cuda'):
        self.seq_len = config['seq_len']
        self.pred_net = RFP(config, device).to(device)
        self.mask_net = Unet(3, 1, config, device=device).to(device)

    def get_params(self):
        return self.pred_net.parameters(), self.mask_net.parameters()

    def eval(self):
        self.pred_net.eval()
        self.mask_net.eval()

    def train(self):
        self.pred_net.train()
        self.mask_net.train()

    def generate_mask(self, inputs):
        mask = self.mask_net(inputs)
        return mask

    def get_prediction(self, inputs, mask, zero_state=None):
        mask_ = mask.repeat_interleave(3, dim=1)
        mask_ = torch.repeat_interleave(mask_.unsqueeze(dim=1), repeats=self.seq_len, dim=1)  # mask_ shape (batch_size, seq_len, dim)
        inputs_mask = torch.mul(inputs, mask_)
        outputs, hx = self.pred_net(inputs_mask, zero_state)
        # outputs, hx = self.pred_net(inputs, zero_state)
        return outputs, hx


          


'''
config = {'input_size': 300, 'hidden_size': [128, 256, 128], 'output_size': 60, 'batch_size': 10, 'period_time': 3371,
          'seq_len': 3371, 'mask': True, 'cell_type': 'GRUCell', 'num_cells': 3, 'same_linear': False, 'device': 'cpu'}
net1 = Mask_Layer(config)

opt = torch.optim.Adam(net1.parameters())
mse = nn.MSELoss(reduce=True, reduction='mean')
def cos_loss(y_pred, y_true):
    return 1 - F.cosine_similarity(y_pred, y_true).abs().mean()


from torch.autograd import Variable
input = torch.randn((config['batch_size'], config['period_time'], config['input_size']), requires_grad = True)
opt.zero_grad()
batch_mask_u = net1(input)
print(batch_mask_u.shape)
y = torch.randn(10, 100)
loss = cos_loss(batch_mask_u, y)

u_dim = Variable(torch.tensor(100))
activate_point = torch.sum(y, dim=1)
loss_2 = torch.pow(torch.div(activate_point, u_dim), 2).mean()
# loss_2 = (activate_point / u_dim) ** 2


loss_total = loss + loss_2
loss_total.backward()
# print(batch_mask_u[0, 0])
opt.step()



batch_mask_u = batch_mask_u.to('cpu')
# print(batch_mask_u.shape)
# print(batch_mask_u[0, 0])
# print(batch_mask_u[1, 0])
'''
