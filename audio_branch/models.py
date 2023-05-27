import torch
import torch.nn as nn

from layers import *

def get_model(configs):
    model = None
    if configs["training"]["model_type"] == "CRNN":
        model =  CRNN(**configs["CRNN"])
    elif configs["training"]["model_type"] == "CNN14":
        model =  Cnn14()
    if configs["training"]["pretrain_dir"] is not None:
        pretrain_dir = configs["training"]["pretrain_dir"]
        checkpoint = torch.load(pretrain_dir)
        model.load_state_dict(checkpoint['model'], strict=False)
    return model


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)
    
class CRNN(nn.Module):
    def __init__(self, nclass=4, activation="glu", dropout=0.2, n_RNN_cell=128, rnn_layers=2, dropout_recurrent=0,  **kwargs):
        super(CRNN, self).__init__()
        self.cnn = CNN(activation=activation, conv_dropout=dropout, **kwargs)
        self.rnn = BidirectionalGRU(n_in=self.cnn.nb_filters[-1], n_hidden=n_RNN_cell, dropout=dropout_recurrent, num_layers=rnn_layers)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(n_RNN_cell * 2, nclass)
        
    def forward(self, x):
        x = x.transpose(1, 2).unsqueeze(1)

        # conv features
        x = self.cnn(x)
        bs, chan, frames, freq = x.size()

        x = x.squeeze(-1)
        x = x.permute(0, 2, 1)  # [bs, frames, chan]

        x = self.rnn(x)
        x = self.dropout(x)

        weak = self.dense(x)  # [bs,  nclass]
        
        return weak.mean(dim=1)


class Cnn14(nn.Module):
    def __init__(self, classes_num=4):
        
        super(Cnn14, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc2 = nn.Linear(2048, classes_num, bias=True)
        self.sf = nn.Softmax(dim=-1)
        
        self.dropout_1 = nn.Dropout(0.2)
        self.dropout_2 = nn.Dropout(0.5)
        self.act = nn.ReLU()
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc2)
 
    def forward(self, x):
        x = x.unsqueeze(1).transpose(2, 3)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.dropout_1(x)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.dropout_1(x)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.dropout_1(x)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = self.dropout_1(x)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = self.dropout_1(x)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = self.dropout_1(x)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = self.dropout_2(x)
        x = self.act(self.fc1(x))
        clipwise_output = self.sf(self.fc2(x))

        return clipwise_output