import torch
import numpy as np
import scipy.io as scio
from torch import nn

@torch.no_grad()
def load_measurement(Path):
    Measurement = scio.loadmat(Path)
    # Establish the pytorch sparse matrix
    indexsparse = Measurement['indexsparse']
    index1 = indexsparse[0,:]
    index2 = indexsparse[1,:]
    indexsparse_value =indexsparse[2,:]
    ##--------------- Original code ---------------##
    i = torch.squeeze(torch.FloatTensor(np.stack([index1, index2], axis=0)).long())
    v = torch.squeeze(torch.FloatTensor(indexsparse_value))
    Aa = torch.sparse_coo_tensor(
        i, v, torch.Size([1000 * 128, 128 * 128])
    ).requires_grad_(False)
    return Aa.cuda()


def get_transform_fn(args):
    im_size = args.image_size
    n = args.len_sig
    m = args.num_sig

    measurement = load_measurement('/storage/data/tongshq/dataset/mice/H_128000_128.mat')

    @torch.no_grad()
    def signal_to_image(signal):
        signal = -1 * signal.reshape(-1, n*m)
        proxy = torch.sparse.mm(measurement.t(), signal.t()).t().reshape(-1, 1, im_size, im_size).transpose(2, 3)
        return proxy

    @torch.no_grad()
    def image_to_signal(image):
        s_hat=image.transpose(2, 3).reshape(-1, im_size*im_size)
        s = torch.sparse.mm(measurement, s_hat.t()).t().reshape(-1, n, m)
        return s
    
    return signal_to_image, image_to_signal


class DAS_operator():
    def __init__(
            self,
            config,
            path='/storage/data/tongshq/dataset/mice/H_128000_128.mat',
            device=torch.device('cuda')
    ):
        super().__init__()
        Measurement = scio.loadmat(path)
        # Establish the pytorch sparse matrix
        indexsparse = Measurement['indexsparse']
        index1 = indexsparse[0,:]
        index2 = indexsparse[1,:]
        indexsparse_value =indexsparse[2,:]
        ##--------------- Original code ---------------##
        i = torch.squeeze(torch.FloatTensor(np.stack([index1, index2], axis=0)).long())
        v = torch.squeeze(torch.FloatTensor(indexsparse_value))
        Aa = torch.sparse_coo_tensor(
            i, v, torch.Size([1000 * 128, 128 * 128])
        ).requires_grad_(False)
        
        self.operator = Aa.to(device)
        self.im_size = config.data.resolution
        self.n = config.data.len_sig
        self.m = config.data.num_sig

    @torch.no_grad()
    def signal_to_image(self, signal):
        signal = -1 * signal.reshape(-1, self.n * self.m)
        proxy = torch.sparse.mm(self.operator.t(), signal.t()).t().reshape(-1, 1, self.im_size, self.im_size).transpose(2, 3)
        return proxy

    @torch.no_grad()
    def image_to_signal(self, image):
        s_hat=image.transpose(2, 3).reshape(-1, self.im_size * self.im_size)
        s = torch.sparse.mm(self.operator, s_hat.t()).t().reshape(-1, self.n, self.m)
        return s