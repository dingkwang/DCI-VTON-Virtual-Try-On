import os
import os.path as osp
import subprocess
import glob
import json
import time
import sys
# Add the path to DensePose to the Python path.
sys.path.append('external/detectron2/projects/DensePose')

from pathlib import Path
import numpy as np
from cv2box import CVImage
from PIL import Image, ImageDraw
from tqdm import tqdm
import fire 

from external.AI_power.seg_lib.cihp_pgn.cihp_pgn_api import load_cihp_model
from external.detectron2.projects.DensePose.apply_net import predict_on_images
from carvekit.web.schemas.config import MLConfig
from carvekit.web.utils.init_utils import init_interface

class Preprocess:
    def __init__(self, model_image_dir) -> None:
        self.model_image_dir = model_image_dir
        self.image_list = glob.glob(osp.join(self.model_image_dir, "*.jpg"))
        self.parent_dir = osp.dirname(osp.normpath(model_image_dir))
        self.openpose_image = osp.join(self.parent_dir, "openpose_image")
        self.openpose_json = osp.join(self.parent_dir, "openpose_json")
        
        self.cihp_model = None
        self.cihp_mask_dir = osp.join(self.parent_dir, "cihp_mask_vis") # For visulization only
        self.cihp_parsing_dir = osp.join(self.parent_dir, "cihp_parsing_maps") # also "image-parse-v3", used in human_agnostic
        self.cihp_edge_dir = osp.join(self.parent_dir, "cihp_edge_maps") # not used. 

        self.densepose_dir = osp.join(self.parent_dir, "image-densepose")

        self.cloth_mask_dir = osp.join(self.parent_dir, "cloth_mask")
        self.image_parse_agnostic_dir = osp.join(self.parent_dir, "image-parse-agnostic-v3.2")

        self.human_agnostic_dir = osp.join(self.parent_dir, "agnostic-v3.2")


    def run_openpose(self):
        """
        Use openpose to generate human pose with hand.
        """
        bin_file = "external/openpose/build/examples/openpose/openpose.bin"
        model_file = "external/openpose/models/pose"
        model_folder = "external/openpose/models"

        assert osp.isfile(bin_file), "openpose bin file doesn't exist."
        assert osp.isdir(model_file), "openpose model file is not downloaded"
            
        os.makedirs(self.openpose_image, exist_ok=True)
        os.makedirs(self.openpose_json, exist_ok=True)
        cmd = f"{bin_file} -model_folder {model_folder} --image_dir {self.model_image_dir}  --hand --disable_blending --display 0 --write_json {self.openpose_json} --write_images {self.openpose_image} --num_gpu 1 --num_gpu_start 0"
        print("run openpose inference\n ", cmd)
        try:
            # Note: shell=True can be a security hazard if untrusted input is passed. Ensure the command is safe and sanitized.
            output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
            print("Console output: \n" + output)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred: \n{str(e)}")
            print(f"Error output: \n{e.output}")

    def run_human_parse(self):
        print("Start human parse")
        start = time.time()
        if self.cihp_model is None:
            self.cihp_model = load_cihp_model()
        print(f"human parse model loading takes{time.time()-start}")
        os.makedirs(self.cihp_mask_dir, exist_ok=True)
        os.makedirs(self.cihp_edge_dir, exist_ok=True)
        os.makedirs(self.cihp_parsing_dir, exist_ok=True)
        for image in self.image_list:
            print("process image", image)
            file_name = osp.basename(image).replace(".jpg", ".png")
            img_p = CVImage(image).bgr
            mask, parsing, edge = self.cihp_model.forward(img_p)
            CVImage(mask).save(osp.join(self.cihp_mask_dir, file_name)) # for vis only
            CVImage(parsing).save(osp.join(self.cihp_parsing_dir, file_name))
            CVImage(edge).save(osp.join(self.cihp_edge_dir, file_name))
        print(f"Human parse done, takes {time.time()-start}")

    def run_densepose(self):
        dense_pose_image_list = [osp.join(self.densepose_dir, osp.basename(image_file)) for image_file in self.image_list]
        predict_on_images(
            config_fpath="external/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml",
            model_fpath="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl",
            image_file_list=self.image_list,
            output_file_list=dense_pose_image_list
        )
        print("Densepose completed.")
        
    def run_clothmask(self):
        """
        #title Upload images from your computer
        #markdown Description of parameters
        #markdown - `SHOW_FULLSIZE`  - Shows image in full size (may take a long time to load)
        #markdown - `PREPROCESSING_METHOD`  - Preprocessing method
        #markdown - `SEGMENTATION_NETWORK`  - Segmentation network. Use `u2net` for hairs-like objects and `tracer_b7` for objects
        #markdown - `POSTPROCESSING_METHOD`  - Postprocessing method
        #markdown - `SEGMENTATION_MASK_SIZE` - Segmentation mask size. Use 640 for Tracer B7 and 320 for U2Net
        #markdown - `TRIMAP_DILATION`  - The size of the offset radius from the object mask in pixels when forming an unknown area
        #markdown - `TRIMAP_EROSION`  - The number of iterations of erosion that the object's mask will be subjected to before forming an unknown area
        """
        if not all((osp.isfile(Path.home() /".cache/carvekit/checkpoints/basnet-universal/basnet.pth"),
                   osp.isfile(Path.home() /".cache/carvekit/checkpoints/deeplabv3-resnet101/deeplab.pth"),
                   osp.isfile(Path.home()/".cache/carvekit/checkpoints/fba/fba_matting.pth"),
                   osp.isfile(Path.home()/ ".cache/carvekit/checkpoints/tracer_b7/tracer_b7.pth"))):
            from carvekit.ml.files.models_loc import download_all
            print("Downloading clothmask models.")
            download_all()

        SHOW_FULLSIZE = False #param {type:"boolean"}
        PREPROCESSING_METHOD = "none" #param ["stub", "none"]
        SEGMENTATION_NETWORK = "tracer_b7" #param ["u2net", "deeplabv3", "basnet", "tracer_b7"]
        POSTPROCESSING_METHOD = "fba" #param ["fba", "none"] 
        SEGMENTATION_MASK_SIZE = 640 #param ["640", "320"] {type:"raw", allow-input: true}
        TRIMAP_DILATION = 30 #param {type:"integer"}
        TRIMAP_EROSION = 5 #param {type:"integer"}
        DEVICE = 'cpu' # 'cuda'
        BACKGROUND = 130

        config = MLConfig(segmentation_network=SEGMENTATION_NETWORK,
                        preprocessing_method=PREPROCESSING_METHOD,
                        postprocessing_method=POSTPROCESSING_METHOD,
                        seg_mask_size=SEGMENTATION_MASK_SIZE,
                        trimap_dilation=TRIMAP_DILATION,
                        trimap_erosion=TRIMAP_EROSION,
                        device=DEVICE)

        interface = init_interface(config)

        os.makedirs(self.cloth_mask_dir, exist_ok=True)
        output_images = interface(self.image_list)
        for image, input_image_file in zip(output_images, self.image_list):
            img = np.array(image)
            img = img[...,:3] # no transparency
            idx = (img[...,0]==BACKGROUND)&(img[...,1]==BACKGROUND)&(img[...,2]==BACKGROUND) # background 0 or 130, just try it
            img = np.ones(idx.shape)*255
            img[idx] = 0
            im = Image.fromarray(np.uint8(img), 'L')
            output_file = osp.join(self.cloth_mask_dir, osp.basename(input_image_file))
            im.save(output_file)

    def parse_agnostic(self):
        """
        Remove cloth parse label and corresponding body parts
        """
        os.makedirs(self.image_parse_agnostic_dir, exist_ok=True)
        for img_file in tqdm(self.image_list):
            # load pose image
            pose_name = osp.basename(img_file).replace('.jpg', '_keypoints.json')
            try:
                with open(osp.join(self.openpose_json, pose_name), 'r') as f:
                    pose_label = json.load(f)
                    pose_data = pose_label['people'][0]['pose_keypoints_2d']
                    pose_data = np.array(pose_data)
                    pose_data = pose_data.reshape((-1, 3))[:, :2]
            except IndexError:
                print(pose_name)
                continue
            # load parsing image
            # parse_name = osp.basename(img_file).replace('.jpg', '.png')
            parse_name = osp.basename(img_file)
            im_parse = Image.open(osp.join(self.cihp_parsing_dir, parse_name)) 
            agnostic_img = get_im_parse_agnostic(im_parse, pose_data)
            agnostic_img.save(osp.join(self.image_parse_agnostic_dir , parse_name))
            # for visualization
            vis_file = osp.join(self.image_parse_agnostic_dir, "vis_"+parse_name)
            visulize_agnostic(agnostic_img, vis_file)

    def human_agnostic(self):
        """
        6. Human Agnostic
        """
        os.makedirs(self.human_agnostic_dir, exist_ok=True)

        for img_file in tqdm(self.image_list):
            # load pose image
            pose_name = osp.basename(img_file).replace('.jpg', '_keypoints.json')
            try:
                with open(osp.join(self.openpose_json, pose_name), 'r') as f:
                    pose_label = json.load(f)
                    pose_data = pose_label['people'][0]['pose_keypoints_2d']
                    pose_data = np.array(pose_data)
                    pose_data = pose_data.reshape((-1, 3))[:, :2]
            except IndexError:
                print(pose_name)
                continue

            # load parsing image
            im = Image.open(img_file)
            label_name = osp.basename(img_file)
            im_label = Image.open(osp.join(self.cihp_parsing_dir, label_name))
            agnostic_img = get_human_agnostic(im, im_label, pose_data)
            agnostic_img.save(osp.join(self.human_agnostic_dir, osp.basename(img_file)))


def get_human_agnostic(img, parse, pose_data):
    parse_array = np.array(parse)
    parse_head = ((parse_array == 4).astype(np.float32) +
                  (parse_array == 13).astype(np.float32))
    parse_lower = ((parse_array == 9).astype(np.float32) +
                   (parse_array == 12).astype(np.float32) +
                   (parse_array == 16).astype(np.float32) +
                   (parse_array == 17).astype(np.float32) +
                   (parse_array == 18).astype(np.float32) +
                   (parse_array == 19).astype(np.float32))

    agnostic = img.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)

    length_a = np.linalg.norm(pose_data[5] - pose_data[2])
    length_b = np.linalg.norm(pose_data[12] - pose_data[9])
    point = (pose_data[9] + pose_data[12]) / 2
    pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
    pose_data[12] = point + (pose_data[12] - point) / length_b * length_a
    r = int(length_a / 16) + 1

    # mask arms
    agnostic_draw.line([tuple(pose_data[i])
                       for i in [2, 5]], 'gray', width=r*10)
    for i in [2, 5]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse(
            (pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')
    for i in [3, 4, 6, 7]:
        if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
            continue
        agnostic_draw.line([tuple(pose_data[j])
                           for j in [i - 1, i]], 'gray', width=r*10)
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse(
            (pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')

    # mask torso
    for i in [9, 12]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse(
            (pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
    agnostic_draw.line([tuple(pose_data[i])
                       for i in [2, 9]], 'gray', width=r*6)
    agnostic_draw.line([tuple(pose_data[i])
                       for i in [5, 12]], 'gray', width=r*6)
    agnostic_draw.line([tuple(pose_data[i])
                       for i in [9, 12]], 'gray', width=r*12)
    agnostic_draw.polygon([tuple(pose_data[i])
                          for i in [2, 5, 12, 9]], 'gray', 'gray')

    # mask neck
    pointx, pointy = pose_data[1]
    agnostic_draw.rectangle(
        (pointx-r*7, pointy-r*7, pointx+r*7, pointy+r*7), 'gray', 'gray')
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic.paste(img, None, Image.fromarray(
        np.uint8(parse_lower * 255), 'L'))
    return agnostic


def get_im_parse_agnostic(im_parse, pose_data, w=768, h=1024):
    label_array = np.array(im_parse)
    parse_upper = ((label_array == 5).astype(np.float32) +
                    (label_array == 6).astype(np.float32) +
                    (label_array == 7).astype(np.float32))
    parse_neck = (label_array == 10).astype(np.float32)

    r = 10
    agnostic = im_parse.copy()

    # mask arms
    for parse_id, pose_ids in [(14, [2, 5, 6, 7]), (15, [5, 2, 3, 4])]:
        mask_arm = Image.new('L', (w, h), 'black')
        mask_arm_draw = ImageDraw.Draw(mask_arm)
        i_prev = pose_ids[0]
        for i in pose_ids[1:]:
            if (pose_data[i_prev, 0] == 0.0 and pose_data[i_prev, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            mask_arm_draw.line([tuple(pose_data[j]) for j in [i_prev, i]], 'white', width=r*10)
            pointx, pointy = pose_data[i]
            radius = r*4 if i == pose_ids[-1] else r*15
            mask_arm_draw.ellipse((pointx-radius, pointy-radius, pointx+radius, pointy+radius), 'white', 'white')
            i_prev = i
        parse_arm = (np.array(mask_arm) / 255) * (label_array == parse_id).astype(np.float32)
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

    # mask torso & neck
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_neck * 255), 'L'))
    return agnostic

def visulize_agnostic(agnostic_img, image_path):
    np_im = np.array(agnostic_img)
    np_im[np_im == 2] = 151
    np_im[np_im == 9] = 178
    np_im[np_im == 13] = 191
    np_im[np_im == 14] = 221
    np_im[np_im == 15] = 246
    image_vis = Image.fromarray(np_im)
    image_vis.save(image_path)


def main(model_image_dir):
    preprocess = Preprocess(model_image_dir=model_image_dir)
    preprocess.run_openpose()
    preprocess.run_human_parse()
    preprocess.run_densepose()
    preprocess.run_clothmask() # Denpends on openpose. 
    preprocess.parse_agnostic()
    preprocess.human_agnostic()



if __name__ == "__main__":
    fire.Fire(main)
