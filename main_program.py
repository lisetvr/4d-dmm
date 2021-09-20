#============================================================
#                 Main script
#  This script contains training and testing functions
#
#  If you use this motion model, please cite our work:
#  "Probabilistic 4D predictive model from in-room surrogates using
#  conditional generative networks for image-guided radiotherapy"
#
#  author: Liset Vazquez Romaguera
#  email: lisetvr90@gmail.com
#  github id: lisetvr
#  MedICAL Lab - Polytechnique Montreal
#============================================================

import warnings
warnings.filterwarnings("ignore")
import argparse
import datetime
from skimage.metrics import structural_similarity as ss
from barbar import Bar
from tensorboardX import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from data_loader import *
from motion_models import *
from utiles import *

parser = argparse.ArgumentParser()

# Path configuration
parser.add_argument('--train_test', type=str, required=True, help='Whether to run training or testing. Options are \"train\" or \"test\".')
parser.add_argument('--data_dir', required=True, help='Path to directory that holds the data.')
parser.add_argument('--logging_dir', required=True, help='Path to logging directory.')
parser.add_argument('--experiment_name', type=str, default='', help='(optional) Name for experiment.')

# Optimization parameters
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--lr_policy', type=str, default='plateau', help='Learning rate scheduler policy. Options are \"linear\", \"step\", \"plateau\" or \"cosine\".')
parser.add_argument('--batch_size', type=int, default=10, help='Batch size.')
parser.add_argument('--max_epoch', type=int, default=400, help='Maximum number of epochs to train for.')
parser.add_argument('--loss_type', type=str, default="3", help='Type of loss. Options are \"1\" for L2 loss (mean), '
                                                               '\"2\" for L2 loss (sum), ''\"3\" for NCCLoss')
parser.add_argument('--recon_weight', type=int, default=1, help='Weight or reconstruction loss.')
parser.add_argument('--KL_weight', type=float, default=0.001, help='Weight for KL loss.')

# Model parameters
parser.add_argument('--condi_type', type=str, default="2", help='Type of conditionant to use. Options are \"0\" for Both, \"1\" for c1, \"2\" for c2')
parser.add_argument('--which_temp_pred', type=str, default="lstm", help='Temporal predictive mechanism. Options are lstm or gru.')
parser.add_argument('--sag_index', type=int, default=16, help='Position of the sagittal slice')
parser.add_argument('--cor_index', type=int, default=32, help='Position of the coronal slice')
parser.add_argument('--exhale_ref', type=bool, default=True, help='Whether to use the exhale position as reference. Uses inhale otherwise')
parser.add_argument('--latent_size', type=int, default=64, help='Latent space dimension')
parser.add_argument('--prelatent_size', type=int, default=16, help='Pre-latent dimension (last FC layer at each branch)')
parser.add_argument('--nb_inputs', type=int, default=3, help='Number of input surrogate images.')
parser.add_argument('--horizon', type=int, default=3, help='Predictive horizon.')
parser.add_argument('--condi_channels', nargs='+', type=int, default=[16, 32, 64], help='Channels in CondiNet.')
parser.add_argument('--enc_channels', nargs='+', type=int, default=[16, 32, 64], help='Channels in encoder.')
parser.add_argument('--dec_channels', nargs='+', type=int, default=[64, 32, 16], help='Channels in decoder.')
parser.add_argument('--vol_size', nargs='+', type=int, default=[32, 64, 64], help='Volume sizes in the dataset.')
# Others
parser.add_argument('--checkpoint', default='', help='Path to a checkpoint to load model weights from.')
parser.add_argument('--VM_checkpoint', default='./VM.pth', help='Path to a Voxelmorph checkpoint to load weights from.')
opt = parser.parse_args()
print('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))
# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(seed=123)
train_folds, valid_folds, test_folds = make_folds()
# -------------------------------- Model creation ----------------------------------------
vm = Voxelmorph(opt.vol_size, [16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16], full_size=True).to(device)
stn = SpatialTransformer(opt.vol_size).to(device)
model = CVAE_model(nb_inputs=opt.nb_inputs, horizon=opt.horizon, vol_size=opt.vol_size, pre_latent_dim=opt.prelatent_size,
                   latent_dim=opt.latent_size, enc_channels=opt.enc_channels, dec_channels=opt.enc_channels,
                   condi_channels=opt.condi_channels, condi_type=opt.condi_type, temp_pred=opt.which_temp_pred).to(device)
# ------------------------------------------------------------------------------------------

# -------------------------------- Loss selection ------------------------------------------
if opt.loss_type == "1":
    criterion = nn.MSELoss(reduction='mean').to(device)
elif opt.loss_type == "2":
    criterion = nn.MSELoss(reduction='sum').to(device)
else:
    criterion = ncc_loss
    criterionDVF = nn.MSELoss(reduction='mean').to(device)
# -------------------------------------------------------------------------------------------

def train(folds=None, fold_idx="0", model=None, dir_name=None):
    if opt.checkpoint:
        custom_load(model, opt.checkpoint, device)
    if opt.VM_checkpoint:
        custom_load(vm, opt.VM_checkpoint, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = get_scheduler(optimizer, opt)
    earlyStopper_recon = EarlyStopping(patience=6, verbose=True, delta=0.01)

    log_dir = os.path.join(opt.logging_dir, 'logs', dir_name, "fold_" + fold_idx)
    run_dir = os.path.join(opt.logging_dir, 'runs', dir_name, "fold_" + fold_idx)
    cond_mkdir(log_dir)
    cond_mkdir(run_dir)

    # Save all command line arguments into a txt file in the logging directory for later reference.
    with open(os.path.join(log_dir, "params.txt"), "w") as out_file:
        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

    writer = SummaryWriter(run_dir)
    train_set = Dataset_4D_multitime(opt.data_dir, sequence_list=folds[0], nb_inputs=opt.nb_inputs, horizon=opt.horizon)
    valid_set = Dataset_4D_multitime(opt.data_dir, sequence_list=folds[1], nb_inputs=opt.nb_inputs, horizon=opt.horizon, valid=True)

    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=4)
    iter = 0
    best_val_loss = np.inf

    print('Begin training...')
    for epoch in range(opt.max_epoch):
        model.train()
        print('Epoch: {}'.format(epoch))
        for ref_volume, input_volume_list, current_volume_list in Bar(train_loader):

            optimizer.zero_grad()
            ref_volume = ref_volume.unsqueeze(1).to(device)
            dvf = []
            for vol in range(len(current_volume_list)):
                current_volume_list[vol] = current_volume_list[vol].unsqueeze(1).to(device)  # For > 1 future volume
                dvf.append(vm(ref_volume, current_volume_list[vol]))

            if opt.condi_type == "1":
                c1 = list()
                c1_ref = ref_volume[:, :, opt.sag_index, :, :]
                for q in range(opt.nb_inputs):
                    c1temp = input_volume_list[q].unsqueeze(1)[:, :, opt.sag_index, :, :]
                    c1.append(torch.cat([c1temp.to(device), c1_ref.to(device)], dim=1))
                c1 = torch.stack(c1, dim=2).to(device)   # sagittal
                c2 = None                                # coronal
            else:
                c2 = list()
                c2_ref = ref_volume[:, :, :, opt.cor_index, :]
                for q in range(opt.nb_inputs):
                    c2temp = input_volume_list[q].unsqueeze(1)[:, :, :, opt.cor_index, :]
                    c2.append(torch.cat([c2temp.to(device), c2_ref.to(device)], dim=1))
                c1 = None                                # sagittal
                c2 = torch.stack(c2, dim=2).to(device)   # coronal

            vmorph_current_volume, kl_loss, generated_dvf, generated_current_volume = model(ref_volume,
                                                                                            current_volume_list,
                                                                                            c1, c2, dvf=dvf)
            kl_loss = opt.KL_weight * kl_loss

            vmorph_recon_loss, recon_loss = 0, 0
            for tt in range(opt.horizon):
                vmorph_recon_loss += criterion(vmorph_current_volume[tt], current_volume_list[tt])
                recon_loss += criterion(generated_current_volume[tt], current_volume_list[tt])

            loss = recon_loss + kl_loss
            loss.backward()
            optimizer.step()

            writer.add_scalar("vmorph_recon_loss", vmorph_recon_loss/opt.horizon, iter)
            writer.add_scalar("recon_loss", recon_loss/opt.horizon, iter)
            writer.add_scalar("kl_loss", kl_loss/opt.horizon, iter)
            writer.add_scalar("total_loss", loss, iter)
            iter += 1

        # Validate model
        optimizer.zero_grad()
        with torch.no_grad():
            model.eval()
            val_loss = 0

            for idx, [ref_volume, input_volume_list, current_volume_list] in enumerate(Bar(valid_loader)):
                ref_volume = ref_volume.unsqueeze(1).to(device)
                dvf = []
                for vol in range(len(current_volume_list)):
                    current_volume_list[vol] = current_volume_list[vol].unsqueeze(1).to(device)  # For > 1 future volume
                    dvf.append(vm(ref_volume, current_volume_list[vol]))

                if opt.condi_type == "1":
                    c1 = list()
                    c1_ref = ref_volume[:, :, opt.sag_index, :, :]
                    for q in range(opt.nb_inputs):
                        c1temp = input_volume_list[q].unsqueeze(1)[:, :, opt.sag_index, :, :]
                        c1.append(torch.cat([c1temp.to(device), c1_ref.to(device)], dim=1))
                    c1 = torch.stack(c1, dim=2).to(device)     # sagittal
                    c2 = None                                  # coronal
                else:
                    c2 = list()
                    c2_ref = ref_volume[:, :, :, opt.cor_index, :]
                    for q in range(opt.nb_inputs):
                        c2temp = input_volume_list[q].unsqueeze(1)[:, :, :, opt.cor_index, :]
                        c2.append(torch.cat([c2temp.to(device), c2_ref.to(device)], dim=1))
                    c1 = None                                 # sagittal
                    c2 = torch.stack(c2, dim=2).to(device)    # coronal

                latent = var_or_cuda(torch.as_tensor(np.random.randn(opt.latent_size), dtype=torch.float), device=device)
                latent = var_or_cuda(latent, device=device).unsqueeze(dim=0)
                generated_dvf, generated_current_volume = model(ref_volume, None, c1, c2, None, prior_post_latent=latent)

                avg_rec_loss = 0
                for tt in range(opt.horizon):
                    avg_rec_loss += criterion(generated_current_volume[tt], current_volume_list[tt]).item()
                val_loss += avg_rec_loss/opt.horizon
            val_loss /= (len(valid_set))

            if val_loss < best_val_loss:
                print("val_loss improved from %0.4f to %0.4f \n" % (best_val_loss, val_loss))
                best_val_loss = val_loss
                custom_save(model, os.path.join(log_dir, 'model_best.pth'))
            else:
                print("val_loss did not improve from %0.4f \n" % (best_val_loss))

            writer.add_scalar("val_loss", val_loss, iter)
            scheduler.step(val_loss)
            earlyStopper_recon(val_loss)

        if earlyStopper_recon.early_stop:
            print("Early stopping")
            break


def test(fold=None, fold_idx="0", dir_name=None):

    with torch.no_grad():
        custom_load(model, os.path.join(opt.checkpoint, "fold_"+fold_idx, "model_best.pth"), device)
        custom_load(vm, opt.VM_checkpoint, device)
        model.eval()

        test_set = Dataset_4D_multitime(opt.data_dir, sequence_list=fold, nb_inputs=opt.nb_inputs, horizon=opt.horizon,
                                        test=True)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

        vol_dir = os.path.join(opt.logging_dir, "test", dir_name, "fold_" + fold_idx, "volumes")
        cond_mkdir(vol_dir)

        MSE_loss, NCC_loss, SSIM_loss = [], [], []
        mse = nn.MSELoss(reduction='mean').to(device)

        for idx, [ref_volume, input_volume_list, current_volume_list] in enumerate(Bar(test_loader)):

            ref_volume = ref_volume.unsqueeze(1).to(device)
            vmorph_volume, dvf = [], []
            for vol in range(len(current_volume_list)):
                current_volume_list[vol] = current_volume_list[vol].unsqueeze(1).to(device)
                dvf_vm = vm(ref_volume, current_volume_list[vol])
                dvf.append(dvf_vm)
                vmorph_volume.append(stn(ref_volume, dvf_vm))

            if opt.condi_type == "1":
                c1 = list()
                c1_ref = ref_volume[:, :, opt.sag_index, :, :]
                for q in range(opt.nb_inputs):
                    c1temp = input_volume_list[q].unsqueeze(1)[:, :, opt.sag_index, :, :]
                    c1.append(torch.cat([c1temp.to(device), c1_ref.to(device)], dim=1))
                c1 = torch.stack(c1, dim=2).to(device)  # sagittal
                c2 = None                               # coronal
            else:
                c2 = list()
                c2_ref = ref_volume[:, :, :, opt.cor_index, :]
                for q in range(opt.nb_inputs):
                    c2temp = input_volume_list[q].unsqueeze(1)[:, :, :, opt.cor_index, :]
                    c2.append(torch.cat([c2temp.to(device), c2_ref.to(device)], dim=1))
                c1 = None                               # sagittal
                c2 = torch.stack(c2, dim=2).to(device)  # coronal

            # Inference
            latent = var_or_cuda(torch.as_tensor(np.random.randn(opt.latent_size), dtype=torch.float), device=device)
            latent = var_or_cuda(latent, device=device).unsqueeze(dim=0)
            generated_dvf, generated_current_volume = model(ref_volume, None, c1, c2, dvf=None, prior_post_latent=latent)

            avg_ncc, avg_mse, avg_ssim = 0, 0, 0
            for tp in range(opt.horizon):
                save_tensor_as_nifti(vmorph_volume[tp][0, 0, :, :, :],
                                     "vm_volume_t" + str(tp),
                                     vol_dir, iter=idx,
                                     aff=[[3.5, 0, 0, 0], [0, 1.70 * 2, 0, 0], [0, 0, 1.70 * 2, 0], [0, 0, 0, 1]])

                save_tensor_as_nifti(generated_current_volume[tp][0, 0, :, :, :],
                                     "generated_volume_t" + str(tp),
                                     vol_dir, iter=idx,
                                     aff=[[3.5, 0, 0, 0], [0, 1.70 * 2, 0, 0], [0, 0, 1.70 * 2, 0], [0, 0, 0, 1]])

                avg_ncc += ncc_loss(generated_current_volume[tp], vmorph_volume[tp], device=device).item()
                avg_mse += mse(generated_current_volume[tp], vmorph_volume[tp]).item()
                avg_ssim += ss(generated_current_volume[tp][0, 0, :, :, :].detach().cpu().numpy(),
                               vmorph_volume[tp][0, 0, :, :, :].detach().cpu().numpy())

            NCC_loss.append(avg_ncc/opt.horizon)
            MSE_loss.append(avg_mse/opt.horizon)
            SSIM_loss.append(avg_ssim/opt.horizon)

        NCC_loss = np.asarray(NCC_loss)
        MSE_loss = np.asarray(MSE_loss)
        SSIM_loss = np.asarray(SSIM_loss)
        dir_name = os.path.join(dir_name, "fold_" + fold_idx)
        np.save(os.path.join(opt.logging_dir, "test", dir_name, "NCC_loss.npy"), NCC_loss)
        np.save(os.path.join(opt.logging_dir, "test", dir_name, "MSE_loss.npy"), MSE_loss)
        np.save(os.path.join(opt.logging_dir, "test", dir_name, "SSIM_loss.npy"), SSIM_loss)

        print("\nTest set average loss NCC: %0.4f, MSE: %0.4f, SSIM: %0.4f" % (np.mean(NCC_loss), np.mean(MSE_loss),
              np.mean(SSIM_loss)))


def main():
    if opt.train_test == "train":
        dir_name = os.path.join(datetime.datetime.now().strftime('%m_%d'),
                                datetime.datetime.now().strftime('%H.%M.%S_') + opt.experiment_name)
        for fold in range(len(train_folds)):
            model = CVAE_model(nb_inputs=opt.nb_inputs, horizon=opt.horizon, vol_size=opt.vol_size,
                               pre_latent_dim=opt.prelatent_size, latent_dim=opt.latent_size,
                               enc_channels=opt.enc_channels, dec_channels=opt.enc_channels,
                               condi_channels=opt.condi_channels, condi_type=opt.condi_type,
                               temp_pred=opt.which_temp_pred).to(device)
            train((train_folds[fold], valid_folds[fold]), fold_idx=str(fold), model=model, dir_name=dir_name)

    elif opt.train_test == "test":
        dir_name = os.path.join(datetime.datetime.now().strftime('%m_%d'),
                                datetime.datetime.now().strftime('%H.%M.%S'))
        for fold in range(len(test_folds)):
            test(test_folds[fold], fold_idx=str(fold), dir_name=dir_name)

    else:
        print("Unknown mode for train_test argument:{}".format(opt.train_test))


if __name__ == "__main__":
    main()