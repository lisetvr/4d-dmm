#============================================================
#                 Deep motion model
#  Implementation of a deep motion model, which is conditioned
#  on 2D cine acquisition and a reference volume
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
from torch.distributions import Normal
from convgru import *
from convlstm import *
from motion_estimation_models import *


class Encoder_CVAE(nn.Module):

    def __init__(self, in_channels, out_channels, output_dim, linear_input_dim=64, norm=nn.BatchNorm3d, dropout=False):

        super().__init__()
        nb_convs = len(out_channels)
        self.encoder = list()
        self.output_dim = output_dim
        self.linear_input_dim = linear_input_dim

        for i in range(nb_convs):
            if i == 0:
                in_ch = in_channels
            else:
                in_ch = out_channels[i - 1]
            self.encoder += [nn.Conv3d(in_ch, out_channels[i], kernel_size=3, padding=1, stride=2)]
            if norm is not None:
                self.encoder += [norm(out_channels[i], affine=True)]
            self.encoder += [nn.ReLU(True)]
            self.encoder += [nn.Conv3d(out_channels[i], out_channels[i], kernel_size=3, padding=1, stride=1)]
            if norm is not None:
                self.encoder += [norm(out_channels[i], affine=True)]
            self.encoder += [nn.ReLU(True)]
            if dropout:
                self.encoder += [nn.Dropout3d()]
        self.encoder = nn.Sequential(*self.encoder)

        self.adap = nn.Conv3d(out_channels[-1], 1, kernel_size=1, stride=1, bias=False)
        nn.init.kaiming_normal_(self.adap.weight, mode='fan_in', nonlinearity='relu')
        self.dvf_enc = nn.Linear(self.linear_input_dim, self.output_dim)

    def forward(self, dvf):
        encoding = self.encoder(dvf)
        encoding = self.adap(encoding).view(-1, self.linear_input_dim)
        encoding = self.dvf_enc(encoding)
        return encoding

class RefCondNet(nn.Module):
    def __init__(self, in_channels, out_channels, output_dim, linear_input_dim=64, norm=nn.BatchNorm3d, dropout=False):

        super().__init__()

        nb_convs = len(out_channels)
        self.encoder = list()
        self.output_dim = output_dim
        self.linear_input_dim = linear_input_dim

        for i in range(nb_convs):
            if i == 0:
                in_ch = in_channels
            else:
                in_ch = out_channels[i - 1]
            self.encoder += [nn.Conv3d(in_ch, out_channels[i], kernel_size=3, padding=1, stride=2)]
            if norm is not None:
                self.encoder += [norm(out_channels[i], affine=True)]
            self.encoder += [nn.ReLU(True)]
            self.encoder += [nn.Conv3d(out_channels[i], out_channels[i], kernel_size=3, padding=1, stride=1)]
            if norm is not None:
                self.encoder += [norm(out_channels[i], affine=True)]
            self.encoder += [nn.ReLU(True)]
            if dropout:
                self.encoder += [nn.Dropout3d()]
        self.encoder = nn.Sequential(*self.encoder)

        self.adap = nn.Conv3d(out_channels[-1], 1, kernel_size=1, stride=1, bias=False)
        nn.init.kaiming_normal_(self.adap.weight, mode='fan_in', nonlinearity='relu')
        self.dvf_enc = nn.Linear(self.linear_input_dim, self.output_dim)

    def forward(self, dvf):
        encoding = self.encoder(dvf)
        encoding = self.adap(encoding).view(-1, self.linear_input_dim)
        encoding = self.dvf_enc(encoding)
        return encoding

class Decoder_CVAE(nn.Module):

    def __init__(self, in_channels, out_channels, z_dim=32+16*2, pre_decoder_dim=64, norm=nn.BatchNorm3d, dropout=False):

        super().__init__()
        nb_convs = len(out_channels)
        self.pre_decoder = nn.Linear(z_dim, pre_decoder_dim)
        self.decoder = list()
        for i in range(nb_convs):
            if i == 0:
                in_ch = in_channels
            else:
                in_ch = out_channels[i - 1]
            self.decoder += [nn.ConvTranspose3d(in_ch, out_channels[i], kernel_size=2, stride=(2, 2, 2))]
            if norm is not None:
                self.decoder += [norm(out_channels[i])]
            self.decoder += [nn.LeakyReLU(0.2)]
            self.decoder += [nn.Conv3d(out_channels[i], out_channels[i], kernel_size=3, padding=1, stride=1)]
            if norm is not None:
                self.decoder += [norm(out_channels[i], affine=True)]
            self.decoder += [nn.LeakyReLU(0.2, True)]
            if dropout:
                self.decoder += [nn.Dropout3d()]
        self.decoder = nn.Sequential(*self.decoder)

        # One conv to get the flow field
        self.flow = nn.Conv3d(out_channels[-1], 3, kernel_size=1, padding=0, stride=1)
        nd = Normal(0, 1e-5) # Make flow weights + bias small. Not sure this is necessary.
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

    def forward(self, feature_and_latent, pre_decoder_shape=(4,4,4)):
        feature_and_latent = self.pre_decoder(feature_and_latent).view(-1, 1, *pre_decoder_shape)
        x = self.decoder(feature_and_latent)
        flow = self.flow(x)
        return flow

class z_generator(nn.Module):
    def __init__(self, input_dim=16*3, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.mu = nn.Linear(self.input_dim, self.latent_dim)
        self.sigma = nn.Linear(self.input_dim, self.latent_dim)

    def forward(self, condis):
        mu = self.mu(condis)
        sigma = self.sigma(condis)

        #reparm trick
        eps = torch.randn_like(mu)
        z = eps.mul(torch.exp(sigma*0.5)).add(mu)
        return z, mu, sigma

class CondiNet(nn.Module):

    def __init__(self, nb_inputs, horizon, in_channels, out_channels, output_dim, condi_type, rnn='lstm'):

        super().__init__()
        nb_convs = len(out_channels)
        self.nb_inputs = nb_inputs
        self.horizon = horizon
        self.output_dim = output_dim
        self.rnn = rnn
        norm = nn.BatchNorm2d
        custom_strides = [2, 2, (1, 2)]
        self.backbone = list()
        for i in range(nb_convs):
            if i == 0:
                in_ch = in_channels
            else:
                in_ch = out_channels[i - 1]

            if condi_type == "1":
                self.backbone += [nn.Conv2d(in_ch, out_channels[i], kernel_size=3, padding=1, stride=2)]
            else:
                self.backbone += [nn.Conv2d(in_ch, out_channels[i], kernel_size=3, padding=1, stride=custom_strides[i])]
            self.backbone += [norm(out_channels[i])]
            self.backbone += [nn.ReLU(True)]
            self.backbone += [nn.Conv2d(out_channels[i], out_channels[i], kernel_size=3, padding=1, stride=1)]
            self.backbone += [norm(out_channels[i])]
            self.backbone += [nn.ReLU(True)]
        self.backbone = nn.Sequential(*self.backbone)

        if rnn == 'gru':
            self.sag_rnn_enc = ConvGRU(in_channels=out_channels[-1], hidden_channels=[out_channels[-1]], kernel_size=(3, 3), num_layers=1, batch_first=True)
            self.sag_rnn_dec = ConvGRU(in_channels=out_channels[-1], hidden_channels=[out_channels[-1]], kernel_size=(3, 3), num_layers=1, batch_first=True)
            self.cor_rnn_enc = ConvGRU(in_channels=out_channels[-1], hidden_channels=[out_channels[-1]], kernel_size=(3, 3), num_layers=1, batch_first=True)
            self.cor_rnn_dec = ConvGRU(in_channels=out_channels[-1], hidden_channels=[out_channels[-1]], kernel_size=(3, 3), num_layers=1, batch_first=True)
        elif rnn == 'lstm':
            self.sag_rnn_enc = ConvLSTM(in_channels=out_channels[-1], hidden_channels=[out_channels[-1]], kernel_size=(3, 3), num_layers=1, batch_first=True)
            self.sag_rnn_dec = ConvLSTM(in_channels=out_channels[-1], hidden_channels=[out_channels[-1]], kernel_size=(3, 3), num_layers=1, batch_first=True)
            self.cor_rnn_enc = ConvLSTM(in_channels=out_channels[-1], hidden_channels=[out_channels[-1]], kernel_size=(3, 3), num_layers=1, batch_first=True)
            self.cor_rnn_dec = ConvLSTM(in_channels=out_channels[-1], hidden_channels=[out_channels[-1]], kernel_size=(3, 3), num_layers=1, batch_first=True)
        else:
            self.cnn = nn.Conv3d(in_channels=out_channels[-1], out_channels=out_channels[-1], kernel_size=(3, 3, 3),
                                 padding=(0, 1, 1), stride=1)
        self.act = nn.ReLU()
        self.norm = nn.BatchNorm3d(out_channels[-1])
        self.adapt = nn.Conv2d(out_channels[-1], 1, kernel_size=1, stride=1, bias=False)
        self.linear = nn.Linear(8*8, output_dim)

    def forward(self, sag, cor):

        if sag is not None:                   # case sagittal
            encsag = []
            for i in range(self.nb_inputs):
                encsag.append(self.backbone(sag[:, :, i, :, :]))
            encsag = torch.stack(encsag, dim=2)
            # --------------------------------------------
            if self.rnn == 'gru':
                encsag = encsag.permute(0, 2, 1, 3, 4)
                decoder_input = self.sag_rnn_enc(encsag)[1][0].unsqueeze(1).repeat(1, self.horizon, 1, 1, 1) # Repeat last state
                states, last_state = self.sag_rnn_dec(decoder_input)
                x = states[0].permute(0, 2, 1, 3, 4)  # (b, t, c, h, w) -> (b, c, t, h, w)
            else:  # lstm
                encsag = encsag.permute(0, 2, 1, 3, 4)
                h_c = self.sag_rnn_enc(encsag)[1][0]
                h, c = h_c[0], h_c[1]
                decoder_input = h.unsqueeze(1).repeat(1, self.horizon, 1, 1, 1)
                states, last_state = self.sag_rnn_dec(decoder_input)
                x = states[0].permute(0, 2, 1, 3, 4)  # (b, t, c, h, w) -> (b, c, t, h, w)
            # --------------------------------------------
            encsag_multitime = self.norm(x)
            encsag_list = []
            for t in range(self.horizon):
                encsag = self.adapt(encsag_multitime[:, :, t, :, :]).view(encsag_multitime.shape[0], -1)
                encsag_list.append(self.linear(encsag))
            return encsag_list

        elif cor is not None:                   # case coronal
            enccor = []
            for i in range(self.nb_inputs):
                enccor.append(self.backbone(cor[:, :, i, :, :]))
            enccor = torch.stack(enccor, dim=2)
            # --------------------------------------------
            if self.rnn == 'gru':
                enccor = enccor.permute(0, 2, 1, 3, 4)
                decoder_input = self.sag_rnn_enc(enccor)[1][0].unsqueeze(1).repeat(1, self.horizon, 1, 1, 1)
                states, last_state = self.cor_rnn_dec(decoder_input)
                x = states[0].permute(0, 2, 1, 3, 4)  # (b, t, c, h, w) -> (b, c, t, h, w)
            else:  # lstm
                enccor = enccor.permute(0, 2, 1, 3, 4)
                h_c = self.cor_rnn_enc(enccor)[1][0]
                h, c = h_c[0], h_c[1]
                decoder_input = h.unsqueeze(1).repeat(1, self.horizon, 1, 1, 1)
                states, last_state = self.cor_rnn_dec(decoder_input)
                x = states[0].permute(0, 2, 1, 3, 4)  # (b, t, c, h, w) -> (b, c, t, h, w)
            # --------------------------------------------
            enccor_multitime = self.norm(x)
            enccor_list = []
            for t in range(self.horizon):
                enccor = self.adapt(enccor_multitime[:, :, t, :, :]).view(enccor_multitime.shape[0], -1)
                enccor_list.append(self.linear(enccor))
            return enccor_list

        else:
            raise NotImplementedError('At least one conditioning plane is required')


class CVAE_model(nn.Module):

    def __init__(self, nb_inputs, horizon, vol_size, pre_latent_dim, latent_dim, enc_channels, dec_channels,
                 condi_channels, condi_type, temp_pred='lstm'):
        """
        Initializes a deep motion model

        Parameters
        ----------
        nb_inputs: int
            Number of input images
        horizon: int
            Predictive horizon
        vol_size: list (int)
            List containing the volume dimensions, e.g [32, 128, 128]
        pre_latent_dim: int
            Hidden dimension of the last fully connected layer of each branch
        latent_dim: int
            Latent size of the motion model
        enc_channels: list (int)
            List containing the channels of each encoder layer
        dec_channels: list (int)
            List containing the channels of each decoder layer
        condi_channels: list (int)
            List containing the channels of each encoder layer in CondiNet
        condi_type: str
            To indicate the orientation of the surrogate images (options: "1" for sagittal, "2" for coronal)
        temp_pred: str
            Type of temporal predictor
        """
        super().__init__()
        self.nb_inputs = nb_inputs
        self.horizon = horizon
        self.pre_latent_dim = pre_latent_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder_CVAE(in_channels=3, out_channels=enc_channels,
                                    norm=nn.BatchNorm3d, output_dim=self.pre_latent_dim, linear_input_dim=4*8*8)
        self.cond_net = CondiNet(nb_inputs=nb_inputs, horizon=horizon, in_channels=2, out_channels=condi_channels,
                                 output_dim=self.pre_latent_dim, condi_type=condi_type, rnn=temp_pred)
        self.ref_net = RefCondNet(in_channels=1, out_channels=enc_channels,
                                  norm=nn.BatchNorm3d, output_dim=self.pre_latent_dim, linear_input_dim=4*8*8)
        self.z_gen = z_generator(input_dim=self.pre_latent_dim * 3, latent_dim=self.latent_dim)
        self.decoder = Decoder_CVAE(in_channels=1, out_channels=dec_channels,
                                    z_dim=self.latent_dim + self.pre_latent_dim * 2, pre_decoder_dim=4*8*8)
        self.spatial_transform = SpatialTransformer(vol_size)

    def forward(self, Vref, Vn, c1, c2, dvf, prior_post_latent=None):

        if Vn is not None:
            vm_vol_list, generated_dvf_list, generated_Vn_list = [], [], []
            kl_loss = 0
            condi_features = self.cond_net(sag=c1, cor=c2)
            condi_features_ref = self.ref_net(Vref)

            for t in range(self.horizon):
                condi_feats = torch.cat((condi_features[t], condi_features_ref), dim=1)
                vm_vol_list.append(self.spatial_transform(Vref, dvf[t]))
                dvf_enc = self.encoder(dvf[t])
                concat = torch.cat((dvf_enc, condi_feats), dim=1)
                z, mu, sigma = self.z_gen(concat)
                z = torch.cat((z,condi_feats), dim=1)
                kl_loss += -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
                generated_dvf = self.decoder(z, pre_decoder_shape=(4,8,8))
                generated_dvf_list.append(generated_dvf)
                generated_Vn_list.append(self.spatial_transform(Vref, generated_dvf))

            return vm_vol_list, kl_loss, generated_dvf_list, generated_Vn_list

        elif prior_post_latent is not None:

            condi_features = self.cond_net(sag=c1, cor=c2)
            condi_features_ref = self.ref_net(Vref)
            generated_dvf_list, generated_Vn_list = [], []

            for t in range(self.horizon):
                condi_feats = torch.cat((condi_features[t], condi_features_ref), dim=1)
                concat = torch.cat((prior_post_latent, condi_feats), dim=1)
                generated_dvf = self.decoder(concat, pre_decoder_shape=(4, 8, 8))
                generated_dvf_list.append(generated_dvf)
                generated_Vn_list.append(self.spatial_transform(Vref, generated_dvf))

            return generated_dvf_list, generated_Vn_list
        else:
            raise NotImplementedError('In CVAE, during inference, the latent should have been sampled from the '
                                      'prior distribution before calling inference.')


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ---------------- Random tensors ---------------------
    vref = torch.randn([2, 1, 32, 64, 64]).to(device)
    vt = torch.randn([2, 1, 32, 64, 64]).to(device)
    sag_images = torch.randn([2, 2, 3, 64, 64]).to(device)
    cor_images = torch.randn([2, 2, 3, 32, 64]).to(device)
    dvf = torch.randn([2, 3, 32, 64, 64]).to(device)
    # ------------------------------------------------------
    model = CVAE_model(nb_inputs=3, horizon=3, vol_size=[32, 64, 64], pre_latent_dim=16, latent_dim=64,
                       enc_channels=[16, 32, 64], dec_channels=[64, 32, 16], condi_channels=[16, 32, 64],
                       condi_type="1", temp_pred='lstm').to(device)
    model_output = model(vref, vt, sag_images, None, dvf=[dvf, dvf, dvf], prior_post_latent=None)
    print()