
from denoising_diffusion_pytorch import Unet, GaussianDiffusion


def define_model():
    model = Unet(

        dim = 128,  # Increased dimensions

        dim_mults = (1, 2, 4, 8, 16),

        channels = 1,  # Assuming grayscale MRI images

        #self_condition = True  # Enable self-conditioning

    )

    return model

def define_diffusion(model):
    diffusion = GaussianDiffusion(

        model = model,

        image_size = 256,

        timesteps = 1000,

        #objective = 'pred_noise',  # Change the objective if needed

        #beta_schedule = 'cosine',  # Changing noise schedule to cosine

    )

    return diffusion







