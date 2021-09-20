import math
import os
import random
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.autograd import Variable
from torch.distributions import kl
from torch.optim import lr_scheduler


def var_or_cuda(x, device=0):
    if torch.cuda.is_available():
        x = x.cuda(device)
    return Variable(x) 

def kl_divergence(posterior_dist, prior_dist, analytic=True):
    """
    Calculate the KL divergence between the posterior and prior KL(Q||P)
    analytic: calculate KL analytically or via sampling from the posterior
    calculate_posterior: if we use samapling to approximate KL we can sample here or supply a sample
    """
    if analytic:
        # Need to add this to torch source code, see: https://github.com/pytorch/pytorch/issues/13545
        result = kl.kl_divergence(posterior_dist, prior_dist)
    else:
        z_posterior = posterior_dist.rsample()
        log_posterior_prob = posterior_dist.log_prob(z_posterior)
        log_prior_prob = prior_dist.log_prob(z_posterior)
        result = log_posterior_prob - log_prior_prob

    return torch.mean(result)

def kld_loss_fn_torch(mean, log_var):
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return KLD

def repeat_cube(tensor, dims):
    s = [1] * len(tensor.size())
    s += dims
    tensor = torch.unsqueeze(tensor, -1)
    tensor = torch.unsqueeze(tensor, -1)
    tensor = torch.unsqueeze(tensor, -1)
    tensor = tensor.repeat(s)
    return tensor

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    From: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    # if opt.lr_policy == 'linear':
    #     def lambda_rule(epoch):
    #         lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
    #         return lr_l
    #     scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    if opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, threshold=0.02, patience=3,
                                                   verbose=True, min_lr=1e-10)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score - self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def custom_load(model, path, device = 'cuda:0'):
    whole_dict = torch.load(path, map_location=device)
    try:
        model.load_state_dict(whole_dict['model'])
    except:
        # If only loading part of the model (e.g Voxelmorph)
        print("No model key in state_dict")
        del whole_dict['spatial_transform.grid']
        model.load_state_dict(whole_dict)

def custom_save(model, path):
    if type(model) == torch.nn.DataParallel:
        model = model.module

    whole_dict = {'model': model.state_dict()}
    torch.save(whole_dict, path)

def save_tensor_as_nifti(tensor, description, val_vol_dir, epoch=-1, iter=-1, aff=np.eye(4)):
    if epoch >= 0:
        # When used during validation
        save_dir = os.path.join(val_vol_dir, 'epoch_%d' % epoch)
    else:
        # When used during testing
        save_dir = os.path.join(val_vol_dir, description)
    cond_mkdir(save_dir)
    # Check if tensor is on gpu
    numpy_tensor = tensor.detach().cpu().numpy()
    if "DVF" in description:
        # Put channel dimension last. Helps with DVF visualization.
        x = np.expand_dims(numpy_tensor[0, :, :, :], -1)
        y = np.expand_dims(numpy_tensor[1, :, :, :], -1)
        z = np.expand_dims(numpy_tensor[2, :, :, :], -1)
        numpy_tensor = np.concatenate([x,y,z], -1)

    new_image = nib.Nifti1Image(numpy_tensor, affine=aff)
    filename = os.path.join(save_dir, (description + '-iter_%05d.nii.gz') % iter)
    nib.save(new_image, filename)

def make_folds():
    case_list = ["CoMoDo01b", "CoMoDo02", "CoMoDo03", "CoMoDo04", "CoMoDo05", "CoMoDo06", "CoMoDo08b", "CoMoDo09",
                 "CoMoDo10", "CoMoDo11", "CoMoDo12", "CoMoDo13", "CoMoDo15", "CoMoDo16", "CoMoDo17", "CoMoDo18",
                 "CoMoDo19", "CoMoDo20", "CoMoDo21", "CoMoDo22", "CoMoDo24", "CoMoDo25", "CoMoDo26", "CoMoDo27",
                 "CoMoDo28"]

    kf = KFold(n_splits=25, shuffle=True, random_state=123)
    kf.get_n_splits(case_list)
    out = kf.split(case_list)
    train = []
    valid = []
    test = []
    random.seed(123)
    for train_idx, test_idx in out:
        train.append(np.take(case_list, train_idx))
        valid_idx = random.sample(list(train_idx), 1)
        valid.append(np.take(case_list, valid_idx))
        train[-1] = np.delete(train[-1], valid_idx)
        test.append(np.take(case_list, test_idx))
    return train, valid, test

# -----------------------------------------------------------------
# ------ Losses ---------------------------------------------------
def ncc_loss(I, J, win=None, device='cuda:0'):
    """
    calculate the normalize cross correlation between I and J
    assumes I, J are sized [batch_size, *vol_shape, nb_feats]
    """

    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    if win is None:
        win = [9] * ndims

    conv_fn = getattr(F, 'conv%dd' % ndims)
    I2 = I * I
    J2 = J * J
    IJ = I * J

    sum_filt = torch.ones([1, 1, *win]).to(device)

    pad_no = math.floor(win[0] / 2)

    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1, 1)
        padding = (pad_no, pad_no)
    else:
        stride = (1, 1, 1)
        padding = (pad_no, pad_no, pad_no)

    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)

    cc = cross * cross / (I_var * J_var + 1e-5)

    return -1 * torch.mean(cc)

def compute_local_sums(I, J, filt, stride, padding, win):
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = F.conv3d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv3d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross

def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
    dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
    dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])

    if(penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    return d / 3.0
# -----------------------------------------------------------------

if __name__ == '__main__':

    train, val, test = make_folds()
    print()
