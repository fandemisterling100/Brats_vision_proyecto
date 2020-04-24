# _*_ coding: utf-8 _*_
# Author: Jielong
# @Time: 21/08/2019 15:42
import sys
import os
import glob
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from utils import normalize, label_converter
np.set_printoptions(threshold=sys.maxsize)


def get_data_paths(data_dir, modality):
    """
    Get get image data paths with corresponding modality (e.g. PET, CT, MASK)
    :param data_dir: the root data directory that contains PET, CT and MASK images
    :param modality: PET/CT/MASK
    :return: data paths
    """
    data_paths = []
    subject_dirs = glob.glob(os.path.join(os.path.dirname(__file__), data_dir, "*"))
    for subject_dir in subject_dirs:
        obj_names = next(os.walk(os.path.join(subject_dir, modality)))[2]
        for fn in obj_names:
            path = os.path.join(subject_dir, modality, fn)
            data_paths.append(path)
    # print(data_paths)
    return data_paths


def data_gen(data_paths, mask_paths):
    """
    get all training images including pet/ct (pet-ct) and mask images
    :param data_paths: data paths for PET images
    :param mask_paths: paths for mask images
    :return: PET images with batch and
    """
    
    no_samples = len(data_paths)
    imgs = np.zeros(shape=(1,no_samples, 155, 240, 240), dtype=np.float32)   # change patch shape if necessary
    mask_imgs = np.zeros(shape=(1,no_samples, 155, 240, 240), dtype=np.float32)
    for i, (img_path, mask_path) in tqdm(enumerate(zip(data_paths, mask_paths)), total=no_samples):
        # print(pet_path)
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        # insert one dimension to the existing data as image channel
        #pet = np.expand_dims(pet, axis=0)
        #mask = np.expand_dims(mask, axis=0)

        # append image
        imgs[i] = img[:,:,0:3]
        mask_imgs[i] = mask[:,:,0:3]

    # Normalize data and convert label value to either 1 or 0
    imgs = imgs / 255.
    mask_imgs = label_converter(mask_imgs)

    print("Loading and Process Complete!")
    return imgs, mask_imgs


def batch_data_gen(imgs, mask_imgs, iter_step, batch_size=3):
    """
    Get training batch to feed convolution neural network
    :param pet_imgs: the whole batch of pet images
    :param mask_imgs: the whole batch of mask images
    :param iter_step: the iteration step during training process
    :param batch_size: batch size to generate
    :return: batch images and batch masks
    """
    
    # shuffling data
    print("entra a batch data gen")
    permutation_idxs = np.random.permutation(len(pet_imgs))
    print("pemutacicion")
    imgs = imgs[permutation_idxs]
    print("Ya adecuacion imagenes")
    mask_imgs = mask_imgs[permutation_idxs]
    print("Ya adecuacion de mascaras ")

    # count iteration step to get corresponding training batch
    step_count = batch_size * iter_step
    print("step_count es:",step_count)
    return imgs[step_count: batch_size + step_count], mask_imgs[step_count: batch_size + step_count]

if __name__ == "__main__":
    import natsort
    data_folder = "processed"
    modalities = ["PET", "MASK"]
    pp = get_data_paths(data_folder, "PET")
    mp = get_data_paths(data_folder, "MASK")
    pp = natsort.natsorted(pp)
    mp = natsort.natsorted(mp)
    print(pp)
    print(mp)
    # pms, mgs = data_gen(pp, mp)
    # print(pms.shape)
    # print(pms[91][0, 0, :, :])
    # print(mgs[91][0, 0, :, :])
    # steps = len(pms) // 16
    # for k in range(steps):
    #     pet_ims, mask_ims = batch_data_gen(pms, mgs, k, batch_size=16)
    #     print(pet_ims.shape)
    # pet_example = sitk.GetArrayFromImage(sitk.ReadImage("data/STS_001/STS_001_PT_COR_16.tiff"))
    # print(pet_example[136, :, :])
