from DDPM import DDPM, ContextUnet
from ViVit import ViViT
from ACDC_loader import ACDCDataset
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

DATA_DIR = "./data"
RESULT_DIR = "./results"

assert os.path.isdir(DATA_DIR), f"{DATA_DIR} is not a directory."
assert os.path.isdir(RESULT_DIR), f"{RESULT_DIR} is not a directory."


def train():
    device = torch.device("cuda")
    n_epoch = 20
    batch_size = 3
    image_size = (32, 128, 128)
    num_frames = 11

    # DDPM hyperparameters
    n_T = 400  # 500
    n_feat = 8  # 128 ok, 256 better (but slower)
    lrate = 1e-4

    # ViViT hyperparameters
    patch_size = (8, 32, 32)

    dataset = ACDCDataset(data_dir="data", split="trn")
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)

    valid_loader = DataLoader(ACDCDataset(data_dir="data", split="val"), batch_size=batch_size, shuffle=False, num_workers=1)
    x_val, x_prev_val = next(iter(valid_loader))
    x_prev_val = x_prev_val.to(device)


    vivit_model = ViViT(image_size, patch_size, num_frames)
    nn_model = ContextUnet(in_channels=1, n_feat=n_feat, in_shape=(1, *image_size))

    ddpm = DDPM(vivit_model=vivit_model, nn_model=nn_model,
                betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(train_loader)
        loss_ema = None
        for x, x_prev in pbar:
            optim.zero_grad()
            x = x.to(device)
            x_prev = x_prev.to(device)
            loss = ddpm(x, x_prev)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        ddpm.eval()
        with torch.no_grad():
            x_gen, x_gen_store = ddpm.sample(x_prev_val, device, guide_w=0.2)
            np.save(f"{RESULT_DIR}/x_gen_{ep}.npy", x_gen)
            np.save(f"{RESULT_DIR}/x_gen_store_{ep}.npy", x_gen_store)


if __name__=="__main__":
    train()