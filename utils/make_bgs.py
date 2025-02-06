import argparse
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from insightface.app import FaceAnalysis
from torchvision.utils import save_image

from model import BiSeNet

device = 'cuda'
checkpoint = './checkpoints'
app = FaceAnalysis(name='antelopev2', root=os.path.join('./', 'third_party_files'),
                       providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))


n_classes = 19
net = BiSeNet(n_classes=n_classes)
net.cuda()
model_pth = './third_party_files/79999_iter.pth'
net.load_state_dict(torch.load(model_pth))
net.eval()




def keep_background(im, parsing_anno, stride):
    # Colors for all 20 parts
    part_colors = [[0, 0, 0], [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0], [0, 0, 0],
                   [0, 0, 0], [0, 0, 0], [0, 0, 0]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8) # [1, 19, 512, 512]

    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)


    for pi in range(1, num_of_class + 1):
        # if pi == 8 or pi == 9 or pi == 14 or pi == 17 or pi == 18:
        #     continue
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    tmp = vis_parsing_anno_color / 255
    mask_channel = 1 - tmp[:, :, 0]
    deep_gray = np.full_like(vis_im, (0, 0, 0), dtype=np.uint8)
    result_image = np.where(mask_channel[:, :, np.newaxis] == 1, 255, deep_gray) # [0-255]
    return result_image


def deal_with_one_image(sorpth, tgtpth, modes):

    trans = transforms.ToTensor()

    to_tensor = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        tgtimg, sorimg = Image.open(tgtpth), Image.open(sorpth)
        tgtimage, sorimage = tgtimg.resize((512, 512), Image.BILINEAR), sorimg.resize((512, 512),
                                                                                      Image.BILINEAR)
        # image = img
        tgtimg, sorimg = to_tensor(tgtimage), to_tensor(sorimage)
        tgtimg, sorimg = torch.unsqueeze(tgtimg, 0), torch.unsqueeze(sorimg, 0)
        tgtimg, sorimg = tgtimg.cuda(), sorimg.cuda()
        tgtout, sorout = net(tgtimg)[0], net(sorimg)[0]  # [1, 19, 512, 512]
        tgtparsing, sorparsing = tgtout.squeeze(0).cpu().numpy().argmax(0), sorout.squeeze(
            0).cpu().numpy().argmax(0)

        tgtbg, sorbg = keep_background(tgtimage, tgtparsing, stride=1), keep_background(sorimage, sorparsing,
                                                                                        stride=1)
        tgtbg, sorbg = cv2.cvtColor(tgtbg, cv2.COLOR_RGB2BGR), cv2.cvtColor(sorbg, cv2.COLOR_RGB2BGR)
        mode_list = modes.split('+')

        if 'pose' in mode_list:
            logical_or = np.bitwise_or(tgtbg, sorbg)
        elif 'light' in mode_list or 'exp' in mode_list:
            logical_or = sorbg
        else:
            raise ValueError(f'Unknown mode: {modes}')

        tmp = logical_or / 255
        mask_channel = 1 - tmp[:, :, 0]
        tmp_sor = cv2.imread(sorpth)
        deep_gray = np.full_like(tmp_sor, (127.5, 127.5, 127.5), dtype=np.uint8)

        im_cv = cv2.cvtColor(tmp_sor, cv2.COLOR_RGB2BGR)
        result_image = np.where(mask_channel[:, :, np.newaxis] == 1, im_cv, deep_gray)
        result_image = trans(result_image)
        # # res = bg + im_pts70
        os.makedirs('results', exist_ok=True)
        save_image(result_image, f"./results/bg_{modes}.png")




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
    deal_with_one_image(args.sor_path, args.tar_path, args.modes)