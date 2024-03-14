import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
from torchvision import transforms


class NiftiiDataset(Dataset):
    def __init__(self, source_paths, target_paths):
        self.source_slices = []
        self.target_slices = []

        for source_path, target_path in zip(source_paths, target_paths):
            source_nii = nib.load(source_path)
            target_nii = nib.load(target_path)

            source_img = torch.tensor(source_nii.get_fdata(dtype=np.float32))
            target_img = torch.tensor(target_nii.get_fdata(dtype=np.float32))

            source_slice = source_img[:, :, source_img.shape[2] // 2].unsqueeze(0)
            target_slice = target_img[:, :, target_img.shape[2] // 2].unsqueeze(0)

            self.source_slices.append(source_slice)
            self.target_slices.append(target_slice)

    def __len__(self):
        return len(self.source_slices)

    def __getitem__(self, idx):
        return self.source_slices[idx], self.target_slices[idx]

def load_data():
    source_image_paths = sorted(glob.glob("/home/youssef/harmo_4/ALL_training_data/Pat*_CHU_zscore_minmax_unbias.nii.gz"))
    target_image_paths = sorted(glob.glob("/home/youssef/harmo_4/ALL_training_data/Pat*_COL_zscore_minmax_unbias.nii.gz"))

    dataset = NiftiiDataset(source_image_paths, target_image_paths)
    dataloader = DataLoader(dataset, batch_size=4)

    return dataloader