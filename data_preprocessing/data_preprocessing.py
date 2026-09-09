import os
import glob
import argparse
import pathlib
from os.path import join
import h5py
import torch
from tqdm import tqdm
from fftc import ifft2c, rss_complex, load_kdata
import numpy as np

def to_tensor(data: np.ndarray) -> torch.Tensor:
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)

if __name__ == '__main__':
    # add argparse
    parser = argparse.ArgumentParser(description='Prepare H5 dataset for CMRxRecon series dataset')
    parser.add_argument('--output_h5_folder', type=str, default='/media/ruru/ad31566c-e032-4ffa-a8cf-751b9dbab424/work/CMRxRecon2025/preprocess',
                        help='path to save H5 dataset')
    parser.add_argument('--input_matlab_folder', type=str,
                        default='/media/ruru/ad31566c-e032-4ffa-a8cf-751b9dbab424/work/CMRxRecon2025/ChallengeData/MultiCoil',
                        help='path to the original matlab data')
    args = parser.parse_args()

    save_folder = args.output_h5_folder
    mat_folder = args.input_matlab_folder

    print('matlab data folder: ', mat_folder)
    print('h5 save folder: ', save_folder)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    print('## step 1: convert matlab training dataset to h5 dataset')

    # file_list = sorted(glob.glob(join(mat_folder, '**/*','FullSample', '**/*', '**/*.mat')))
    file_list = sorted(glob.glob(join(mat_folder, 'TrainingSet/FullSample', '**/*', '**/*.mat')))
    print('number of total matlab files: ', len(file_list))

    # check if cuda is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    for ff in tqdm(file_list):
        ##* get info from path — derive relative path from the input root
        save_name = os.path.splitext(os.path.relpath(ff, mat_folder))[0] + '.h5'
        save_name_file = join(save_folder, save_name)
        if not os.path.exists(pathlib.Path(save_name_file).parent):
            os.makedirs(pathlib.Path(save_name_file).parent)
        ##* load kdata
        kdata = load_kdata(ff)
        if 'T1w' in ff or 'T2w' in ff or 'blackblood' in ff:
            kdata = np.expand_dims(kdata, -1)
            kdata = kdata.transpose(4, 3, 2, 1, 0)
        if 'Center007' in ff:
            kdata = kdata.transpose(4, 3, 2, 1, 0)

        ##* swap phase_encoding and readout
        kdata = kdata.swapaxes(-1, -2)

        ##* get rss from kdata
        kdata_th = to_tensor(kdata)
        img_coil = ifft2c(kdata_th).to(device)
        img_rss = rss_complex(img_coil, dim=-3).cpu().numpy()

        ##* save h5
        file = h5py.File(save_name_file, 'w')
        file.create_dataset('kspace', data=kdata)
        file.create_dataset('reconstruction_rss', data=img_rss)

        file.attrs['max'] = img_rss.max()
        file.attrs['norm'] = np.linalg.norm(img_rss)
        file.attrs['shape'] = kdata.shape
        file.attrs['padding_left'] = 0
        file.attrs['padding_right'] = kdata.shape[-1]
        file.attrs['encoding_size'] = (kdata.shape[-2], kdata.shape[-1], 1)
        file.attrs['recon_size'] = (kdata.shape[-2], kdata.shape[-1], 1)
        file.attrs['patient_id'] = save_name
        file.close()
