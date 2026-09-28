# -*- coding: utf-8 -*-
import numpy as np
import pgml
import torch

from popgenml.data.simulators import MSPrimeSimulator
from popgenml.data.functions import graph_to_tree, tree_to_FW, distmat_to_tree
from popgenml.data.relate import relate

class UNetRegressor(object):
    def __init__(self, model, scales, mu_x, std_x, mu_y, std_y, wavelet = 'shannon', 
                         w0 = 1.0, L = 2.5e6, stride = 512, size = 1024, device = 'cuda'):
        self.model = model
        self.scales = scales
        self.device = torch.device(device)
        
        if wavelet == "shannon":
            self.config = pgml.WaveletConfig.shannon()
        elif wavelet == "gaussian":
            self.config = pgml.WaveletConfig.gaussian(order = w0)
        else:
            self.config = pgml.WaveletConfig.morlet(w0 = w0)
        
        self.mu = mu_x
        self.std = std_x
        
        self.mu_y = mu_y
        self.std_y = std_y
        
        self.size = size
        self.stride = stride
        
    def predict(self, x, pos, mask_channel=None):        
        start = 0
        end = self.size
        
        D_pred = np.zeros((x.shape[1], x.shape[0] * (x.shape[0] - 1) // 2))
        count = np.zeros(D_pred.shape)
        
        windows = []
        while end < x.shape[-1]:
            windows.append((start, end))
            
            start += self.stride
            end += self.stride
            
        windows.append((x.shape[-1] - self.size, x.shape[-1]))
        
        # get the input: mean centered wavelet coefficients + alignment + diffed positions
        pos_ = np.diff(pos, append = np.zeros(1))
        xw = (pgml.compute_cwt(x, self.scales, self.config) - self.mu.reshape(self.mu.shape[0], 1, 1)) / self.std.reshape(self.mu.shape[0], 1, 1)
        
        x_input = np.concatenate([np.expand_dims(x, 0), xw])
        x_input = np.concatenate([x_input, np.tile(pos_.reshape(1, 1, -1), (1, x_input.shape[1], 1))], 0)

        # APPLY OCCLUSION MASK
        if mask_channel is not None:
            # Set the entire masked channel to 0 (the mean value post-normalization)
            x_input[mask_channel, :, :] = 0.0

        for ix, window in enumerate(windows):                    
            a, b = window
            xw_window = x_input[:,:,a:b]
            
            xw_window = torch.FloatTensor(xw_window).to(self.device).unsqueeze(0)
            
            with torch.no_grad():
                D_pred_, xe = self.model(xw_window)
                D_pred_ = D_pred_.detach().cpu().numpy()
            
            D_pred_ = D_pred_[0]
            
            D_pred[a:b] += np.exp(D_pred_ * self.std_y + self.mu_y)
            count[a:b] += 1.
            
        # final distance prediction
        D_pred = D_pred / count

        return D_pred        
        
    
