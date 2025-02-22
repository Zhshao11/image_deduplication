import os
import sys
import time
import shutil
# import cv2
# import matplotlib.pyplot as plt
import numpy as np
# from six import moves
# from torch.signal.windows import gaussian

import omniglue.src.omniglue
# # from DetectorFreeSfM.third_party.aspantransformer.tools.preprocess_scene import image
# from omniglue.src.omniglue import utils
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
def match(image0_path,image1_path,sp_feature0,sp_feature1,dino_descriptor0,dino_descriptor1):
    image0 = np.array(Image.open(image0_path).convert("RGB"))
    image1 = np.array(Image.open(image1_path).convert("RGB"))
    start = time.time()
    print(f"> \tTook {time.time() - start} seconds.")
    # Perform inference.
    print("> Finding matches...")
    start = time.time()
    match_kp0, match_kp1, match_confidences = og.FindMatches(image0, image1,sp_feature0,sp_feature1,dino_descriptor0,dino_descriptor1)
    num_matches = match_kp0.shape[0]
    print(f"> \tFound {num_matches} matches.")
    print(f"> \tTook {time.time() - start} seconds.")
    # Filter by confidence (0.02).
    print("> Filtering matches...")
    match_threshold = 0.02  # Choose any value [0.0, 1.0).
    keep_idx = []
    for i in range(match_kp0.shape[0]):
        if match_confidences[i] > match_threshold:
            keep_idx.append(i)
    num_filtered_matches = len(keep_idx)
    print(f"> \tFound {num_filtered_matches}/{num_matches} above threshold {match_threshold}")
    return num_filtered_matches/num_matches
def calculate_feature(image_path):
    image = np.array(Image.open(image_path).convert("RGB"))
    sp_feature,dino_descriptor = og.extract_features(image)
    return sp_feature,dino_descriptor
if __name__ == "__main__":
    image_files = []
    image_files.append([f for f in os.listdir(sys.argv[1]) if f.endswith(('.jpg', '.jpeg', '.png', 'bmp', '.tiff'))])#返回图片名称
    move = 0
    features = {}
    og = omniglue.src.omniglue.OmniGlue(
        og_export="./models/og_export",
        sp_export="./models/sp_v6",
        dino_export="./models/dinov2_vitb14_pretrain.pth",
    )
    for image_file in image_files[0][:]:
        image_path = os.path.join(sys.argv[1], image_file)
        sp_feature,dino_descriptor = calculate_feature(image_path)
        features[image_file] = {'sp_feature':sp_feature,'dino_descriptor':dino_descriptor}
    while True:
        image_files = []
        image_files.append(
            [f for f in os.listdir(sys.argv[1]) if f.endswith(('.jpg', '.jpeg', '.png', 'bmp', '.tiff'))])  # 返回图片名称
        if image_files[0]:
            pass
        else:break
        move_path = "/sda/zhushihao/datasets/AirBnB/Airbnb_Data/Test_Data/image_dudup/"+str(move)+'/'
        os.mkdir(move_path)
        image0_file = image_files[0][0]
        image0_path = os.path.join(sys.argv[1], image_files[0][0])
        for image_file in image_files[0][1:]:
            image1_path = os.path.join(sys.argv[1], image_file)
            sp_feature0 = features[image0_file]['sp_feature']
            sp_feature1 = features[image_file]['sp_feature']
            dino_descriptor0 = features[image0_file]['dino_descriptor']
            dino_descriptor1 = features[image_file]['dino_descriptor']
            rate = match(image0_path,image1_path,sp_feature0,sp_feature1,dino_descriptor0,dino_descriptor1)
            if rate > 0.6:
                print(f"匹配图像0为：",{image0_path})
                print(f"匹配图像1为：",{image1_path})
                try:
                    shutil.move(image1_path,move_path)
                    move +=1
                except:pass
        shutil.move(image0_path, move_path)
        move += 1

