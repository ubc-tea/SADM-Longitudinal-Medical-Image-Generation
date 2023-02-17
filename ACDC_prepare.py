import os
import nibabel as nib
from tqdm import tqdm
from glob import glob
from scipy.ndimage import zoom
import numpy as np


DATA_DIR = "./data/"
assert os.path.isdir(DATA_DIR), f"{DATA_DIR} is not a directory."


def download_and_unzip():
    zip_dir = os.path.join(DATA_DIR, "database.zip")
    url = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/folder/637218e573e9f0047faa00fc/download"

    if not os.path.exists(zip_dir):
        print("Downloading ACDC dataset... (about 2GB)")
        os.system(f"curl -L {url} -o {zip_dir}")
    if not os.path.exists(os.path.join(DATA_DIR, "database/testing/patient150/patient150_4d.nii.gz")):
        print("Unzipping ACDC dataset...")
        os.system(f"unzip -o {zip_dir} -d {DATA_DIR}")

def preprocess():
    patient_fns = glob(os.path.join(DATA_DIR, "database", "*/patient*/"))

    resize_shape = (128,128,32,12)
    all_dat = np.zeros(shape=(len(patient_fns),)+resize_shape, dtype=np.float32)

    for pat_fn in tqdm(patient_fns):
        pat_idx = int(pat_fn.split("/patient")[-1][:-1])
        with open(pat_fn+"Info.cfg", "r") as f:
            ED_idx = int(f.readline().split("ED:")[-1])
            ES_idx = int(f.readline().split("ES:")[-1])

        dat = nib.load(pat_fn + "patient%03d_4d.nii.gz"%pat_idx).get_fdata()[...,ED_idx-1:ES_idx]
        all_dat[pat_idx-1] = zoom(dat, zoom=np.divide(resize_shape,dat.shape))



    all_dat = np.transpose(all_dat , (0,4,1,2,3))

    all_dat -= np.amin(all_dat, axis=(2,3,4), keepdims=True)
    all_dat /= np.amax(all_dat, axis=(2,3,4), keepdims=True)

    np.random.seed(0)

    rand_idx = np.random.permutation(100)

    trn_dat = all_dat[rand_idx[:90]]
    val_dat = all_dat[rand_idx[90:]]
    tst_dat = all_dat[100:]

    np.save(os.path.join(DATA_DIR, "trn_dat.npy"), trn_dat)
    np.save(os.path.join(DATA_DIR, "val_dat.npy"), val_dat)
    np.save(os.path.join(DATA_DIR, "tst_dat.npy"), tst_dat)

if __name__ == "__main__":
    download_and_unzip()
    preprocess()