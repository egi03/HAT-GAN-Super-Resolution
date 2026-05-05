import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from torch.utils import data as data

from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

from basicsr.degradation_machine import DegradationMachine

@DATASET_REGISTRY.register(suffix='basicsr')
class RealESRGANDataset(data.Dataset):
    """Dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(RealESRGANDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']

        # file client (lmdb io backend)
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError(f"'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            # disk backend with meta_info
            # Each line in the meta_info describes the relative path to an image
            with open(self.opt['meta_info']) as fin:
                paths = [line.strip().split(' ')[0] for line in fin]
                self.paths = [os.path.join(self.gt_folder, v) for v in paths]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        
        # Robust loading loop to handle corrupt images
        while True:
            retry = 3
            while retry > 0:
                try:
                    img_bytes = self.file_client.get(gt_path, 'gt')
                except (IOError, OSError) as e:
                    logger = get_root_logger()
                    logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                    index = random.randint(0, self.__len__() - 1)
                    gt_path = self.paths[index]
                    time.sleep(1)
                else:
                    break
                finally:
                    retry -= 1
            
            try:
                img_gt = imfrombytes(img_bytes, float32=True)
                break # Image loaded successfully
            except AttributeError:
                # Occurs if imfrombytes fails to decode and returns None (then .astype fails)
                logger = get_root_logger()
                logger.warn(f'Corrupt image found at: {gt_path}. Resampling...')
                index = random.randint(0, self.__len__() - 1)
                gt_path = self.paths[index]

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])

        # -------------------- Crop ili Pad na fiksnu veličinu -------------------- #
        # Standardno je 400x400 za GT, što nakon 4x smanjenja daje 100x100 LR
        h, w = img_gt.shape[0:2]
        crop_pad_size = self.opt.get('gt_size', 400) # Uzimamo iz .yml datoteke
        
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            h, w = img_gt.shape[0:2]
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]

        # -------------------------------------------------------------------------------- #
        # INTEGRACIJA TVOG STROJA ZA KVARENJE
        # -------------------------------------------------------------------------------- #
        
        # 1. Priprema za stroj (iz float32 [0,1] u uint8 [0,255])
        img_gt_uint8 = (img_gt * 255.).astype(np.uint8)
        
        # 2. Inicijalizacija i pokretanje tvog stroja
        # Napomena: Za maksimalnu brzinu, degrader možeš inicijalizirati u __init__ klase
        degrader = DegradationMachine()
        img_lq_uint8 = degrader.run(img_gt_uint8) # Tvoj "Ultimate" pipeline

        # 3. Konverzija natrag u Tensor format za PyTorch
        # BGR to RGB nije nužno ako tvoj degrader već radi u BGR (OpenCV standard)
        img_gt_tensor = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        
        # LQ slika mora biti float32 i normalizirana na [0, 1]
        img_lq = img_lq_uint8.astype(np.float32) / 255.
        img_lq_tensor = img2tensor([img_lq], bgr2rgb=True, float32=True)[0]

        # -------------------------------------------------------------------------------- #
        # Povrat podataka modelu
        # -------------------------------------------------------------------------------- #
        
        return {
            'lq': img_lq_tensor,       # Degradirana slika (ulaz u HAT)
            'gt': img_gt_tensor,       # Originalna slika (cilj/target)
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)
