#============================================================
#                Data loader for volumes
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

import os
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset


class Dataset_4D_multitime(Dataset):

    def __init__(self, root_dir, sequence_list, nb_inputs=3, horizon=1, valid=False, test=False):
        """
        Initialize a Dataset class

        Parameters
        ----------
        root_dir: str
            Path to the folder that contains the dataset
        sequence_list: list (str)
            List of the subject's Ids, e.g. ['CoMoDo01b', 'CoMoDo02', ...]
        nb_inputs: int
            Number of input images
        horizon: int
            Predictive horizon
        valid: bool
            To indicate when used for validation
        test: bool
            To indicate when used for testing
        """
        self.root_dir = root_dir
        self.nb_inputs = nb_inputs
        self.horizon = horizon
        self.files_input = list()
        self.files_output = list()
        self.ref_vol_files = {}
        self.exhale_as_reference = True
        self.test = test
        self.valid = valid
        # Annotations of reference (inhale and exhale) respiratory phases for each subject
        self.ref_id = {'CoMoDo01b': (4, 9), 'CoMoDo02': (19, 26), 'CoMoDo03': (12, 7), 'CoMoDo04': (24, 2), 'CoMoDo05': (7, 1),
                       'CoMoDo06': (22, 9), 'CoMoDo08b': (10, 19), 'CoMoDo09': (6, 18), 'CoMoDo10': (27, 10), 'CoMoDo11': (15, 0),
                       'CoMoDo12': (4, 23), 'CoMoDo13': (7, 14), 'CoMoDo15': (8, 23), 'CoMoDo16': (4, 20), 'CoMoDo17': (18, 3),
                       'CoMoDo18': (9, 18), 'CoMoDo19': (11, 22), 'CoMoDo20': (17, 13), 'CoMoDo21': (10, 5), 'CoMoDo22': (8, 18),
                       'CoMoDo24': (4, 12), 'CoMoDo25': (14, 19), 'CoMoDo26': (12, 6), 'CoMoDo27': (9, 5), 'CoMoDo28': (6, 19)}
        if self.test:
            temp_navs = 20
        elif self.valid:
            temp_navs = 20
        else:
            temp_navs = 80

        for sequence in sequence_list:
            if sequence == ".directory":
                continue
            data_dir = os.path.join(self.root_dir, sequence)

            for z in range(temp_navs):
                for t in range(31 - (self.nb_inputs + self.horizon)):
                    img = list()
                    label = list()
                    for i in range(self.nb_inputs):
                        img.append(data_dir + '/t_' + str((31 * z + t) + i) + '.nii.gz')
                    for p in range(1, horizon + 1):
                        label.append(data_dir + '/t_' + str((31 * z + t) + (self.nb_inputs - 1) + p) + '.nii.gz')
                    self.files_input.append(img)
                    self.files_output.append(label)

            volume_files = os.listdir(data_dir)
            volume_files.sort(key=lambda x: int(x[2:-7]))
            (inh, exh) = self.ref_id[sequence]

            if self.exhale_as_reference:
                ref_phase = exh
            else:
                ref_phase = inh

            ref_vol_file = [file for file in volume_files if file.endswith("t_" + str(ref_phase) + ".nii.gz")][0]

            if not self.test:  # When testing we want to make sure that the model does not move the reference volume
                self.files_input = [[ele for ele in sub if ele != ref_phase] for sub in self.files_input]
                indices = []
                for (ind, value) in enumerate(self.files_input):
                    if len(value) != self.nb_inputs:
                        indices.append(ind)
                self.files_input = [i for i in self.files_input if len(i) == self.nb_inputs]
                self.files_output = [j for (i, j) in enumerate(self.files_output) if i not in indices]

            self.ref_vol_files[sequence] = ("{}/{}".format(data_dir, ref_vol_file))


    def __len__(self):
        return len(self.files_input)

    def __getitem__(self, idx):
        # Load input volumes
        input_volume_list = list()
        for vol_file in self.files_input[idx]:
            input_volume = nib.load(vol_file).get_fdata()
            input_volume = (torch.from_numpy(input_volume)).float().unsqueeze(0).unsqueeze(0)
            input_volume = F.interpolate(input_volume, scale_factor=[1, 0.5, 0.5], mode='trilinear').squeeze()
            input_volume = (input_volume - torch.mean(input_volume)) / torch.std(input_volume)
            input_volume_list.append(input_volume)

        # Load output volume
        output_volume_list = list()
        for vol_file in self.files_output[idx]:
            output_volume = nib.load(vol_file).get_fdata()
            output_volume = (torch.from_numpy(output_volume)).float().unsqueeze(0).unsqueeze(0)
            output_volume = F.interpolate(output_volume, scale_factor=[1, 0.5, 0.5], mode='trilinear').squeeze()
            output_volume = (output_volume - torch.mean(output_volume)) / torch.std(output_volume)
            output_volume_list.append(output_volume)

        # Load reference volume
        ref_vol_file = self.ref_vol_files[self.files_input[idx][0].split('/')[-2]]
        ref_volume = nib.load(ref_vol_file).get_fdata()
        ref_volume = (torch.from_numpy(ref_volume)).float().unsqueeze(0).unsqueeze(0)
        ref_volume = F.interpolate(ref_volume, scale_factor=[1, 0.5, 0.5], mode='trilinear').squeeze()
        ref_volume = (ref_volume - torch.mean(ref_volume)) / torch.std(ref_volume)

        return ref_volume, input_volume_list, output_volume_list



