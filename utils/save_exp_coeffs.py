
import argparse
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from insightface.app import FaceAnalysis

import datasets_faceswap as datasets_faceswap
import third_party.d3dfr.bfm as bfm
import third_party.model_resnet_d3dfr as model_resnet_d3dfr

device = 'cpu'
checkpoint = './checkpoints'

app = FaceAnalysis(name='antelopev2', root=os.path.join('./', 'third_party_files'),
                       providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

app.prepare(ctx_id=0, det_size=(640, 640))

pil2tensor = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)])


net_d3dfr = model_resnet_d3dfr.getd3dfr_res50(os.path.join(checkpoint, 'third_party/d3dfr_res50_nofc.pth')).eval().to(device)

bfm_facemodel = bfm.BFM(focal=1015*256/224, image_size=256,
                            bfm_model_path=os.path.join(checkpoint, 'third_party/BFM_model_front.mat')).to(device)


def get_landmarks(image):
    face_info = app.get(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
    if len(face_info) == 0:
        return 'error'
    face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * x['bbox'][3] - x['bbox'][1])[-1]
    pts5 = face_info['kps']

    warp_mat = datasets_faceswap.get_affine_transform(pts5, datasets_faceswap.mean_face_lm5p_256)
    drive_im_crop256 = cv2.warpAffine(np.array(image), warp_mat, (256, 256), flags=cv2.INTER_LINEAR)

    drive_im_crop256_pil = Image.fromarray(drive_im_crop256)
    image_tar_crop256 = pil2tensor(drive_im_crop256_pil).view(1, 3, 256, 256)

    gt_d3d_coeff = net_d3dfr(image_tar_crop256)
    # _, ex_coeff = bfm_facemodel.get_lm68(gt_d3d_coeff)
    id_coeff, ex_coeff, tex_coeff, angles, gamma, translation = bfm_facemodel.split_coeff_orderly(gt_d3d_coeff)

    return (ex_coeff, angles, gamma, translation)



def main(sorpth, tarpth, modes):

    mode_list = modes.split('+')
    if 'exp' in mode_list:
        dstpth = tarpth
    elif 'light' in mode_list or 'pose' in mode_list:
        dstpth = sorpth
    else:
        raise ValueError('Unrecognized mode')
    with torch.no_grad():
        img = Image.open(dstpth)
        res = get_landmarks(img)
        if isinstance(res, str):
            print('cannot find face on ', dstpth)
            return
        ex_coeff, angles, gamma, translation = res
        os.makedirs('results', exist_ok=True)
        np.save(f"./results/exp_{modes}.npy", ex_coeff[0])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sor_path",
        type=str,
        default='',
        required=False
    )
    parser.add_argument(
        "--tar_path",
        type=str,
        default='',
        required=False
    )
    parser.add_argument(
        "--modes",
        type=str,
        default='',
        required=False
    )

    args = parser.parse_args()
    main(args.sor_path, args.tar_path, args.modes)