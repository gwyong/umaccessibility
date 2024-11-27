import numpy as np
import cv2
from utils import *
import pandas as pd
from metrics import *
import matplotlib.pyplot as plt 
from scipy.stats import pearsonr

# Define the color list
color_list = {0: [255, 0, 0], 1: [0, 255, 0], 2: [0, 0, 255]}

# Image dimensions
img_width = 640
img_height = 640

# Function to convert normalized YOLO coordinates to pixel coordinates
def yolo_to_pixel_coords(coords, img_width, img_height):
    return [(int(x * img_width), int(y * img_height)) for x, y in zip(coords[::2], coords[1::2])]

def processing_one(input_file_path):
    with open(input_file_path, "r") as file:
        lines = file.readlines()
    output_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        normalized_coords = list(map(float, parts[1:]))
        pixel_coords = yolo_to_pixel_coords(normalized_coords, img_width, img_height)
        polygon = np.array(pixel_coords, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(output_img, [polygon], color_list[class_id])
    return output_img

def generate_unlabeled_groundtruth():
    base_path = 'data_200/initial/labels/unlabeled'
    output_path = 'data_200/initial/true_mask/unlabeled'
    txt_list = get_files_list(base_path, '.txt')
    for i in range(len(txt_list)):
        processed_img = processing_one(base_path + '/' + txt_list[i])
        output_file_path = output_path + '/' + txt_list[i][:-4] + '.png'
        cv2.imwrite(output_file_path, processed_img)
#generate_unlabeled_groundtruth()

def cyclewise_evaluation():
    cycle = '3'
    groundtruth_path = 'data_200/initial/true_mask/unlabeled'
    inferenced_path = 'data_200/cycle' + cycle + '/inferenced/result_mask'
    score_df = pd.read_csv('data_200_result/cycle' + cycle + '/chatgpt_evaluation.csv')
    filename_list = score_df['filename']
    score_list = score_df['score']

    iou_total_list = []
    score_total_list = []
    for i in range(len(filename_list)):
        score_here = score_list[i]
        groundtruth_mask = cv2.imread(groundtruth_path + '/' + filename_list[i])
        inferenced_mask = cv2.imread(inferenced_path + '/' + filename_list[i])
        iou_list = []
        for j in range(3):
            groundtruth_mask_layer = groundtruth_mask[:,:,j]/255
            inferenced_mask_layer = inferenced_mask[:,:,j]/255
            pixel_list = calculate_metrics(groundtruth_mask_layer, inferenced_mask_layer)
            # [TP, FN, FP, TN]
            TP, FN, FP, TN = pixel_list
            if (TP+FN+FP) != 0:
                iou = TP/(TP+FP+FN)
                iou_list.append(iou)
        score_total_list.append(int(score_here))
        if len(iou_list) == 0 : 
            iou_total_list.append(-1)
        else:
            iou_total_list.append(np.mean(iou_list))
    r, _ = pearsonr(iou_total_list, score_total_list)
    print("correlation:", r)
    plt.scatter(iou_total_list, score_total_list, s = 1)
    plt.xlim(0,1)
    plt.xlabel("iou (calculated by ground truth label)")
    plt.ylabel("ChatGPT's score")
    plt.title("Relationship between iou and ChatGPT's score")
    plt.show()
    

cyclewise_evaluation()
