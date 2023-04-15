import torch
import torch.nn as nn
import tqdm
import numpy as np
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa import LogmelFilterBank, Spectrogram

from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

class Feature_extractor(nn.Module):
    def __init__(self, configs, train_loader=None):
        super(Feature_extractor, self).__init__()
        
        self.configs = configs
        feat_params = configs["feats"]
        
        self.mel_spec = MelSpectrogram(
        sample_rate=feat_params["sr"],
        n_fft=feat_params["n_window"],
        win_length=feat_params["n_window"],
        hop_length=feat_params["hop_length"],
        f_min=feat_params["f_min"],
        f_max=feat_params["f_max"],
        n_mels=feat_params["n_mels"],
        window_fn=torch.hamming_window,
        wkwargs={"periodic": False},
        power=1
        )
        
        self.amp_to_db = AmplitudeToDB(stype="amplitude")
        self.amp_to_db.amin = 1e-10
#         self.mel_spec = Spectrogram(n_fft=feat_params["n_window"],
#                                      hop_length=feat_params["hop_length"],
#                                      win_length=feat_params["n_window"],
#                                      window='hann',
#                                      center=True,
#                                      pad_mode='reflect',
#                                      freeze_parameters=True)

#         self.amp_to_db = LogmelFilterBank(sr=feat_params["sr"],
#                                           n_fft=feat_params["n_window"],
#                                           n_mels=feat_params["n_mels"],
#                                           fmin=feat_params["f_min"],
#                                           fmax=feat_params["f_max"],
#                                           ref=1.0,
#                                           amin=1e-10,
#                                           top_db=None,
#                                           freeze_parameters=True)
        if train_loader is not None:
            self.scaler = TorchScaler("instance", "minmax", [1, 2])
            self.scaler.fit(
                train_loader,
                transform_func=lambda x: self.amp_to_db(self.mel_spec(x[1])).clamp(min=-50, max=80),
            )
            # torch.save(self.scaler, configs["training"]["scaler_path"])
        else:
            self.scaler = torch.load(configs["training"]["scaler_path"])

    def forward(self, x):
        tmp = self.mel_spec(x)
        tmp = self.amp_to_db(tmp).clamp(min=-50, max=80)
        return self.scaler(tmp)

def mixup(data, labels):
    batch_size = data.size(0)
    c = np.random.beta(0.2, 0.2)
    perm = torch.randperm(batch_size)
    mixed_mels = c * data + (1 - c) * data[perm, :]
    return mixed_mels, c, perm

def filt_aug(features, db_range=[-6, 6], n_band=[3, 6], min_bw=6):
    batch_size, n_freq_bin, _ = features.shape
    n_freq_band = torch.randint(low=n_band[0], high=n_band[1], size=(1,)).item()   # [low, high)
    if n_freq_band > 1:
        while n_freq_bin - n_freq_band * min_bw + 1 < 0:
            min_bw -= 1
        band_bndry_freqs = torch.sort(torch.randint(0, n_freq_bin - n_freq_band * min_bw + 1,
                                                    (n_freq_band - 1,)))[0] + \
                           torch.arange(1, n_freq_band) * min_bw
        band_bndry_freqs = torch.cat((torch.tensor([0]), band_bndry_freqs, torch.tensor([n_freq_bin])))
        band_factors = torch.rand((batch_size, n_freq_band + 1)).to(features) * (db_range[1] - db_range[0]) + db_range[0]
        freq_filt = torch.ones((batch_size, n_freq_bin, 1)).to(features)
        for i in range(n_freq_band):
            for j in range(batch_size):
                freq_filt[j, band_bndry_freqs[i]:band_bndry_freqs[i+1], :] = \
                    torch.linspace(band_factors[j, i], band_factors[j, i+1],
                                   band_bndry_freqs[i+1] - band_bndry_freqs[i]).unsqueeze(-1)
        freq_filt = 10 ** (freq_filt / 20)
        return features * freq_filt
    else:
        return features

def spec_aug(features):
    tmp_features = features.unsqueeze(1).transpose(2, 3)
    spec_augmenter = SpecAugmentation(time_drop_width=48, time_stripes_num=2, freq_drop_width=4, freq_stripes_num=2)
    tmp_features = spec_augmenter(tmp_features)
    tmp_features = tmp_features.squeeze(1).transpose(1, 2)
    return tmp_features

class TorchScaler(nn.Module):
    def __init__(self, statistic="dataset", normtype="standard", dims=(1, 2), eps=1e-8):
        super(TorchScaler, self).__init__()
        assert statistic in ["dataset", "instance", None]
        assert normtype in ["standard", "mean", "minmax", None]
        if statistic == "dataset" and normtype == "minmax":
            raise NotImplementedError(
                "statistic==dataset and normtype==minmax is not currently implemented."
            )
        self.statistic = statistic
        self.normtype = normtype
        self.dims = dims
        self.eps = eps

    def load_state_dict(self, state_dict, strict=True):
        if self.statistic == "dataset":
            super(TorchScaler, self).load_state_dict(state_dict, strict)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if self.statistic == "dataset":
            super(TorchScaler, self)._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )

    def fit(self, dataloader, transform_func=lambda x: x[0]):
        indx = 0
        for batch in tqdm.tqdm(dataloader):

            feats = transform_func(batch)
            if indx == 0:
                mean = torch.mean(feats, self.dims, keepdim=True).mean(0).unsqueeze(0)
                mean_squared = (
                    torch.mean(feats ** 2, self.dims, keepdim=True).mean(0).unsqueeze(0)
                )
            else:
                mean += torch.mean(feats, self.dims, keepdim=True).mean(0).unsqueeze(0)
                mean_squared += (
                    torch.mean(feats ** 2, self.dims, keepdim=True).mean(0).unsqueeze(0)
                )
            indx += 1

        mean /= indx
        mean_squared /= indx

        self.register_buffer("mean", mean)
        self.register_buffer("mean_squared", mean_squared)

    def forward(self, tensor):

        if self.statistic is None or self.normtype is None:
            return tensor

        if self.statistic == "dataset":
            assert hasattr(self, "mean") and hasattr(
                self, "mean_squared"
            ), "TorchScaler should be fit before used if statistics=dataset"
            assert tensor.ndim == self.mean.ndim, "Pre-computed statistics "
            if self.normtype == "mean":
                return tensor - self.mean
            elif self.normtype == "standard":
                std = torch.sqrt(self.mean_squared - self.mean ** 2)
                return (tensor - self.mean) / (std + self.eps)
            else:
                raise NotImplementedError

        else:
            if self.normtype == "mean":
                return tensor - torch.mean(tensor, self.dims, keepdim=True)
            elif self.normtype == "standard":
                return (tensor - torch.mean(tensor, self.dims, keepdim=True)) / (
                    torch.std(tensor, self.dims, keepdim=True) + self.eps
                )
            elif self.normtype == "minmax":
                return (tensor - torch.amin(tensor, dim=self.dims, keepdim=True)) / (
                    torch.amax(tensor, dim=self.dims, keepdim=True)
                    - torch.amin(tensor, dim=self.dims, keepdim=True)
                    + self.eps
                )
