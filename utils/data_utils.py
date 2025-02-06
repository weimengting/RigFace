


import os

import numpy as np
import scipy
import scipy.io
import torch
from skimage.io import imread
from skimage.transform import estimate_transform, warp, resize

from decalib.datasets import detectors

face_detector = detectors.FAN()
scale = 1.3
resolution_inp = 224



def bbox2point(left, right, top, bottom, type='bbox'):

    if type =='kpt68':
        old_size = (right - left + bottom - top) / 2 * 1.1
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
    elif type =='bbox':
        old_size = (right - left + bottom - top ) /2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size *0.12])
    else:
        raise NotImplementedError
    return old_size, center





def get_image_dict(img_path, size, iscrop):
    img_name = img_path.split('/')[-1]
    im = imread(img_path)
    if size is not None:  # size = 256
        im = (resize(im, (size, size), anti_aliasing=True) * 255.).astype(np.uint8)
    # (256, 256, 3)
    image = np.array(im)
    if len(image.shape) == 2:
        image = image[:, :, None].repeat(1, 1, 3)
    if len(image.shape) == 3 and image.shape[2] > 3:
        image = image[:, :, :3]

    h, w, _ = image.shape
    if iscrop:  # true
        # provide kpt as txt file, or mat file (for AFLW2000)
        kpt_matpath = os.path.splitext(img_path)[0] + '.mat'
        kpt_txtpath = os.path.splitext(img_path)[0] + '.txt'
        if os.path.exists(kpt_matpath):
            kpt = scipy.io.loadmat(kpt_matpath)['pt3d_68'].T
            left = np.min(kpt[:, 0])
            right = np.max(kpt[:, 0])
            top = np.min(kpt[:, 1])
            bottom = np.max(kpt[:, 1])
            old_size, center = bbox2point(left, right, top, bottom, type='kpt68')
        elif os.path.exists(kpt_txtpath):
            kpt = np.loadtxt(kpt_txtpath)
            left = np.min(kpt[:, 0])
            right = np.max(kpt[:, 0])
            top = np.min(kpt[:, 1])
            bottom = np.max(kpt[:, 1])
            old_size, center = bbox2point(left, right, top, bottom, type='kpt68')
        else:
            bbox, bbox_type = face_detector.run(image)
            if len(bbox) < 4:
                print('no face detected! run original image')
                left = 0
                right = h - 1
                top = 0
                bottom = w - 1
            else:
                left = bbox[0]
                right = bbox[2]
                top = bbox[1]
                bottom = bbox[3]
            old_size, center = bbox2point(left, right, top, bottom, type=bbox_type)
        size = int(old_size * scale)
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
    else:
        src_pts = np.array([[0, 0], [0, h - 1], [w - 1, 0]])
    # DST_PTS = np.array([[0, 0], [0, h-1], [w-1, 0]])
    DST_PTS = np.array([[0, 0], [0, resolution_inp - 1], [resolution_inp - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    image = image / 255.

    dst_image = warp(image, tform.inverse, output_shape=(resolution_inp, resolution_inp))
    dst_image = dst_image.transpose(2, 0, 1)
    return {'image': torch.tensor(dst_image).float(),
            'imagename': img_name,
            'tform': torch.tensor(tform.params).float(),
            'original_image': torch.tensor(image.transpose(2, 0, 1)).float(),
            }
