from data_load import load_data
from model import define_model
from model import define_diffusion
from training import train_model
import torch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = load_data()
    model = define_model().to(device)
    diffusion = define_diffusion(model)
    train_model(dataloader, model, diffusion)


if __name__ == "__main__":
    main()



