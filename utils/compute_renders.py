
import os
import sys

import argparse
import torch as th
from torchvision.utils import save_image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))


from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg
from data_utils import get_image_dict

# Build DECA
deca_cfg.model.use_tex = True
deca_cfg.model.tex_path = "./data/FLAME_texture.npz"
deca_cfg.model.tex_type = "FLAME"
deca = DECA(config=deca_cfg, device="cuda")



def get_render(source, target, modes):
    src_dict = get_image_dict(source, 512, True)
    tar_dict = get_image_dict(target, 512, True)
    # ===================get DECA codes of the target image===============================
    tar_cropped = tar_dict["image"].unsqueeze(0).to("cuda")
    imgname = tar_dict["imagename"]
    with th.no_grad():
        tar_code = deca.encode(tar_cropped)
    tar_image = tar_dict["original_image"].unsqueeze(0).to("cuda")
    # ===================get DECA codes of the source image===============================
    src_cropped = src_dict["image"].unsqueeze(0).to("cuda")
    with th.no_grad():
        src_code = deca.encode(src_cropped)
    # To align the face when the pose is changing
    src_ffhq_center = deca.decode(src_code, return_ffhq_center=True)
    tar_ffhq_center = deca.decode(tar_code, return_ffhq_center=True)

    src_tform = src_dict["tform"].unsqueeze(0)
    src_tform = th.inverse(src_tform).transpose(1, 2).to("cuda")
    src_code["tform"] = src_tform

    tar_tform = tar_dict["tform"].unsqueeze(0)
    tar_tform = th.inverse(tar_tform).transpose(1, 2).to("cuda")
    tar_code["tform"] = tar_tform

    src_image = src_dict["original_image"].unsqueeze(0).to("cuda")  # 平均的参数
    tar_image = tar_dict["original_image"].unsqueeze(0).to("cuda")

    # code 1 means source code, code 2 means target code
    code1, code2 = {}, {}
    for k in src_code:
        code1[k] = src_code[k].clone()

    for k in tar_code:
        code2[k] = tar_code[k].clone()

    mode_list = modes.split("+")
    # 应该是确定有pose参与，就转换目标为target
    if 'pose' in mode_list:
        if 'exp' not in mode_list:
            code2['exp'] = src_code['exp']
            code2['pose'][:, 3:] = src_code['pose'][:, 3:]
        if 'light' not in mode_list:
            code2['light'] = src_code['light']
        opdict, _ = deca.decode(
            code2,
            render_orig=True,
            original_image=tar_image,
            tform=tar_code["tform"],
            align_ffhq=True,
            ffhq_center=tar_ffhq_center,
        )
    else:
        if 'exp' in mode_list:
            code1['exp'] = tar_code['exp']
            code1['pose'][:, 3:] = tar_code['pose'][:, 3:]
        if 'light' not in mode_list:
            code1['light'] = tar_code['light']
        opdict, _ = deca.decode(
            code1,
            render_orig=True,
            original_image=src_image,
            tform=src_code["tform"],
            align_ffhq=True,
            ffhq_center=src_ffhq_center,
        )

    rendered = opdict["rendered_images"].detach()
    os.makedirs('results', exist_ok=True)
    save_image(rendered[0], f"./results/render_{modes}.png")



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
    get_render(args.sor_path, args.tar_path, args.modes)
    print('done')