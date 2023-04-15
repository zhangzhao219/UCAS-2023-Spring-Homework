import torch
import torch.nn as nn

from audio_layers import *

def get_model(configs):
    model = None
    if configs["training"]["model_type"] == "CRNN":
        model =  CRNN()
    elif configs["training"]["model_type"] == "CNN14":
        model =  Cnn14()
    elif configs["training"]["model_type"] == "CNN10":
        model = Cnn10()
    elif configs["training"]["model_type"] == "HTSAT":
        model = HTSAT_Swin_Transformer()
    if configs["training"]["pretrain_dir"] is not None:
        if configs["training"]["model_type"] == "HTSAT":
            retrain_part = ["head.weight", "head.bias", "tscam_conv.weight", "tscam_conv.bias"]
            if configs["training"]["pretrain_dir"] is not None:
                pretrain_dir = configs["training"]["pretrain_dir"]
                ckpt = torch.load(pretrain_dir, map_location="cpu")
                ckpt = ckpt["state_dict"]
                # load pretrain model
                found_parameters, unfound_parameters = [], []
                model_params = dict(model.state_dict())

                for key in model_params:
                    m_key = "sed_model." + key
                    if m_key in ckpt:
                        if m_key == "sed_model.patch_embed.proj.weight":
                            ckpt[m_key] = torch.mean(ckpt[m_key], dim = 1, keepdim = True)
                        if key in retrain_part:
                            ckpt.pop(m_key)
                            unfound_parameters.append(key)
                            continue
                        found_parameters.append(key)
                        ckpt[key] = ckpt.pop(m_key)
                    else:
                        unfound_parameters.append(key)
                model.load_state_dict(ckpt, strict = False)
        else:            
            pretrain_dir = configs["training"]["pretrain_dir"]
            checkpoint = torch.load(pretrain_dir, map_location=torch.device('cpu'))
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
    def __init__(self, nclass=4, activation="glu", dropout=0.2, n_RNN_cell=128, rnn_layers=2, dropout_recurrent=0,  kernel_size=[3, 3, 3, 3, 3, 3, 3], padding=[1, 1, 1, 1, 1, 1, 1], stride=[1, 1, 1, 1, 1, 1, 1], nb_filters=[ 4, 8, 16, 16, 16, 32, 32], pooling=[ [ 2, 2 ], [ 2, 2 ], [ 1, 2 ], [ 1, 2 ], [ 1, 2 ], [ 1, 2 ], [ 1, 2 ] ] ):
        super(CRNN, self).__init__()
        self.cnn = CNN(activation=activation, conv_dropout=dropout, kernel_size=kernel_size, padding=padding, stride=stride, nb_filters=nb_filters, pooling=pooling)
        self.rnn = BidirectionalGRU(n_in=self.cnn.nb_filters[-1], n_hidden=n_RNN_cell, dropout=dropout_recurrent, num_layers=rnn_layers)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(n_RNN_cell * 2, nclass)
        self.sf = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = x.transpose(1, 2).unsqueeze(1)

        # conv features
        x = self.cnn(x)
        bs, chan, frames, freq = x.size()

        x = x.squeeze(-1)
        x = x.permute(0, 2, 1)  # [bs, frames, chan]

        x, _ = self.rnn(x)
        x1 = x.mean(1)
        x2 = x.max(1)[0]
        x = x1 + x2
        x= self.dropout(x)

        x = self.dense(x)  # [bs,  nclass]
        
        return self.sf(x)


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
 
    def forward(self, x, return_feat=False, return_fr_feat=False):
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
        fr_feat = x.transpose(1, 2)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = self.dropout_2(x)
        x = self.act(self.fc1(x))
        feat = x
        clipwise_output = self.sf(self.fc2(x))
        
        if return_feat:
            return feat
        elif return_fr_feat:
            return fr_feat
        else:
            return clipwise_output

class Cnn10(nn.Module):
    def __init__(self, classes_num=4):
        
        super(Cnn10, self).__init__()

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

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc2 = nn.Linear(512, classes_num, bias=True)
        self.sf = nn.Softmax(dim=-1)
        
        self.dropout_1 = nn.Dropout(0.2)
        self.dropout_2 = nn.Dropout(0.5)
        self.act = nn.ReLU()
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc2)
 
    def forward(self, x, return_feat=False, return_fr_feat=False):
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
        x = torch.mean(x, dim=3)
        fr_feat = x.transpose(1, 2)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = self.dropout_2(x)
        x = self.act(self.fc1(x))
        feat = x
        clipwise_output = self.sf(self.fc2(x))
        
        if return_feat:
            return feat
        elif return_fr_feat:
            return fr_feat
        else:
            return clipwise_output

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

    def extra_repr(self):
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


# We use the model based on Swintransformer Block, therefore we can use the swin-transformer pretrained model
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_before_mlp='ln'):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm_before_mlp = norm_before_mlp
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.norm_before_mlp == 'ln':
            self.norm2 = nn.LayerNorm(dim)
        elif self.norm_before_mlp == 'bn':
            self.norm2 = lambda x: nn.BatchNorm1d(dim)(x.transpose(1, 2)).transpose(1, 2)
        else:
            raise NotImplementedError
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        # pdb.set_trace()
        H, W = self.input_resolution
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows, attn = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, attn

    def extra_repr(self):
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"



class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self):
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 norm_before_mlp='ln'):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer, norm_before_mlp=norm_before_mlp)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        attns = []
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x, attn = blk(x)
                if not self.training:
                    attns.append(attn.unsqueeze(0))
        if self.downsample is not None:
            x = self.downsample(x)
        if not self.training:
            attn = torch.cat(attns, dim = 0)
            attn = torch.mean(attn, dim = 0)
        return x, attn

    def extra_repr(self):
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


# The Core of HTSAT
class HTSAT_Swin_Transformer(nn.Module):
    def __init__(self, spec_size=256, patch_size=4, patch_stride=(4,4), 
                in_chans=1, num_classes=4,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[4, 8, 16, 32],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, 
                 ape=False, patch_norm=True,
                 use_checkpoint=False, norm_before_mlp='ln',**kwargs):
        super(HTSAT_Swin_Transformer, self).__init__()

        self.spec_size = spec_size 
        self.patch_stride = patch_stride
        self.patch_size = patch_size
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.ape = ape
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = len(self.depths)
        self.num_features = int(self.embed_dim * 2 ** (self.num_layers - 1))
        self.enable_repeat_mode = False
        
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.enable_tscam = True

        self.qkv_bias = qkv_bias
        self.qk_scale = None

        self.patch_norm = patch_norm
        self.norm_layer = norm_layer if self.patch_norm else None
        self.norm_before_mlp = norm_before_mlp
        self.mlp_ratio = mlp_ratio

        self.use_checkpoint = use_checkpoint

        #  process mel-spec ; used only once
        self.freq_ratio = self.spec_size // 64
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.interpolate_ratio = 32     # Downsampled ratio
        self.bn0 = nn.BatchNorm2d(64)


        # split spctrogram into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=self.spec_size, patch_size=self.patch_size, in_chans=self.in_chans, 
            embed_dim=self.embed_dim, norm_layer=self.norm_layer, patch_stride = patch_stride)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.grid_size
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=self.drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(self.embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                    patches_resolution[1] // (2 ** i_layer)),
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                drop=self.drop_rate, attn_drop=self.attn_drop_rate,
                drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                norm_layer=self.norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                norm_before_mlp=self.norm_before_mlp)
            self.layers.append(layer)

        self.norm = self.norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)

        if self.enable_tscam:
            SF = self.spec_size // (2 ** (len(self.depths) - 1)) // self.patch_stride[0] // self.freq_ratio
            self.tscam_conv = nn.Conv2d(
                in_channels = self.num_features,
                out_channels = self.num_classes,
                kernel_size = (SF,3),
                padding = (0,1)
            )
            self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        else:
            self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.sf = nn.Softmax(dim=-1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}


    def forward_features(self, x, return_feat=False):
        frames_num = x.shape[2]        
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for i, layer in enumerate(self.layers):
            x, attn = layer(x)

        if self.enable_tscam:
            # for x
            x = self.norm(x)
            B, N, C = x.shape
            SF = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[0]
            ST = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[1]
            x = x.permute(0,2,1).contiguous().reshape(B, C, SF, ST)
            B, C, F, T = x.shape
            # group 2D CNN
            c_freq_bin = F // self.freq_ratio
            x = x.reshape(B, C, F // c_freq_bin, c_freq_bin, T)
            x = x.permute(0,1,3,2,4).contiguous().reshape(B, C, c_freq_bin, -1)

            # get latent_output
            latent_output = self.avgpool(torch.flatten(x,2))
            latent_output = torch.flatten(latent_output, 1)   
            
            if return_feat:
                return latent_output
            else:
                return self.sf(self.head(latent_output))

    def crop_wav(self, x, crop_size, spe_pos = None):
        time_steps = x.shape[2]
        tx = torch.zeros(x.shape[0], x.shape[1], crop_size, x.shape[3]).to(x.device)
        for i in range(len(x)):
            if spe_pos is None:
                crop_pos = random.randint(0, time_steps - crop_size - 1)
            else:
                crop_pos = spe_pos
            tx[i][0] = x[i, 0, crop_pos:crop_pos + crop_size,:]
        return tx

    # Reshape the wavform to a img size, if you want to use the pretrained swin transformer model
    def reshape_wav2img(self, x):
        B, C, T, F = x.shape
        target_T = int(self.spec_size * self.freq_ratio)
        target_F = self.spec_size // self.freq_ratio
        assert T <= target_T and F <= target_F, "the wav size should less than or equal to the swin input size"
        # to avoid bicubic zero error
        if T < target_T:
            x = nn.functional.interpolate(x, (target_T, x.shape[3]), mode="bicubic", align_corners=True)
        if F < target_F:
            x = nn.functional.interpolate(x, (x.shape[2], target_F), mode="bicubic", align_corners=True)
        x = x.permute(0,1,3,2).contiguous()
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], self.freq_ratio, x.shape[3] // self.freq_ratio)
        x = x.permute(0,1,3,2,4).contiguous()
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3], x.shape[4])
        return x
    
    # Repeat the wavform to a img size, if you want to use the pretrained swin transformer model
    def repeat_wat2img(self, x, cur_pos):
        B, C, T, F = x.shape
        target_T = int(self.spec_size * self.freq_ratio)
        target_F = self.spec_size // self.freq_ratio
        assert T <= target_T and F <= target_F, "the wav size should less than or equal to the swin input size"
        # to avoid bicubic zero error
        if T < target_T:
            x = nn.functional.interpolate(x, (target_T, x.shape[3]), mode="bicubic", align_corners=True)
        if F < target_F:
            x = nn.functional.interpolate(x, (x.shape[2], target_F), mode="bicubic", align_corners=True)  
        x = x.permute(0,1,3,2).contiguous() # B C F T
        x = x[:,:,:,cur_pos:cur_pos + self.spec_size]
        x = x.repeat(repeats = (1,1,4,1))
        return x

    def forward(self, x: torch.Tensor, mixup_lambda = None, infer_mode = False):# out_feat_keys: List[str] = None):
        x = x.unsqueeze(1).transpose(2, 3)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if infer_mode:
            frame_num = x.shape[2]
            target_T = int(self.spec_size * self.freq_ratio)
            repeat_ratio = math.floor(target_T / frame_num)
            x = x.repeat(repeats=(1,1,repeat_ratio,1))
            x = self.reshape_wav2img(x)
            output_dict = self.forward_features(x)
        elif self.enable_repeat_mode:
            if self.training:
                cur_pos = random.randint(0, (self.freq_ratio - 1) * self.spec_size - 1)
                x = self.repeat_wat2img(x, cur_pos)
                output_dict = self.forward_features(x)
            else:
                output_dicts = []
                for cur_pos in range(0, (self.freq_ratio - 1) * self.spec_size + 1, self.spec_size):
                    tx = x.clone()
                    tx = self.repeat_wat2img(tx, cur_pos)
                    output_dicts.append(self.forward_features(tx))
                clipwise_output = torch.zeros_like(output_dicts[0]["clipwise_output"]).float().to(x.device)
                framewise_output = torch.zeros_like(output_dicts[0]["framewise_output"]).float().to(x.device)
                for d in output_dicts:
                    clipwise_output += d["clipwise_output"]
                    framewise_output += d["framewise_output"]
                clipwise_output  = clipwise_output / len(output_dicts)
                framewise_output = framewise_output / len(output_dicts)

                output_dict = {
                    'framewise_output': framewise_output, 
                    'clipwise_output': clipwise_output
                }
        else:
            if x.shape[2] > self.freq_ratio * self.spec_size:
                if self.training:
                    x = self.crop_wav(x, crop_size=self.freq_ratio * self.spec_size)
                    x = self.reshape_wav2img(x)
                    output_dict = self.forward_features(x)
                else:
                    # Change: Hard code here
                    overlap_size = (x.shape[2] - 1) // 4
                    output_dicts = []
                    crop_size = (x.shape[2] - 1) // 2
                    for cur_pos in range(0, x.shape[2] - crop_size - 1, overlap_size):
                        tx = self.crop_wav(x, crop_size = crop_size, spe_pos = cur_pos)
                        tx = self.reshape_wav2img(tx)
                        output_dicts.append(self.forward_features(tx))
                    clipwise_output = torch.zeros_like(output_dicts[0]["clipwise_output"]).float().to(x.device)
                    framewise_output = torch.zeros_like(output_dicts[0]["framewise_output"]).float().to(x.device)
                    for d in output_dicts:
                        clipwise_output += d["clipwise_output"]
                        framewise_output += d["framewise_output"]
                    clipwise_output  = clipwise_output / len(output_dicts)
                    framewise_output = framewise_output / len(output_dicts)
                    output_dict = {
                        'framewise_output': framewise_output, 
                        'clipwise_output': clipwise_output
                    }
            else: # this part is typically used, and most easy one
                x = self.reshape_wav2img(x)
                output_dict = self.forward_features(x)
        return output_dict
