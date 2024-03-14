#this script takes a functio that takes the input images and the saved model after training and return a harmonized image
from model import define_model
import torch
from torchvision.utils import save_image
import nibabel as nib
import numpy as np 
from data_load import NiftiiDataset, load_data


def generate_harmonized_image(model_path, source_image_path):
    # Load the trained model
    model = define_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load the source image
    source_nii = nib.load(source_image_path)
    source_img = torch.tensor(source_nii.get_fdata(dtype=np.float32))
    source_slice = source_img[:, :, source_img.shape[2] // 2].unsqueeze(0).unsqueeze(0)

    # Generate a random timestep
    timesteps = 1000
    t = torch.randint(0, timesteps, (source_slice.size(0),), device=source_slice.device)

    # Generate the harmonized image
    with torch.no_grad():
        harmonized_slice = model(source_slice, t)

    # Save the harmonized image
    save_image(harmonized_slice, '/home/youssef/harmo_4/harmonized_result/harmonized_pat_42.png')

source_image_path = "/home/youssef/harmo_4/ALL_training_data/Pat42_CHU_zscore_minmax_unbias.nii.gz"
model_path = "/home/youssef/harmo_4/trained_model/model_checkpoint_8_gpu.pth"

generate_harmonized_image(model_path, source_image_path)

