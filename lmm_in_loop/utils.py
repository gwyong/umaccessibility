import os
import glob
import cv2
import numpy as np
import json
import re
import base64

def get_files_list(folder_path, extension):
    file_list = []
    for file in os.listdir(folder_path):
        if file.endswith(extension):
            file_list.append(file)
    return file_list

def exclude_elements(a_list, b_list):
    return [item for item in a_list if item not in b_list]

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)  # 디렉토리 생성
        print(f"directory made: {directory_path}")
    else:
        print(f"directory exist: {directory_path}")

def labelme_to_rgb_mask(json_file, image_shape):
    label_color_map = {
            'crosswalk': [255,0,0], 
            'curbcut': [0,255,0], 
            'sidewalk': [0,0,255], 
        }
    with open(json_file) as f:
        data = json.load(f)
    mask = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    for shape in data['shapes']:
        label = shape['label']
        if shape['shape_type'] == 'polygon' and label in label_color_map:
            points = np.array(shape['points'], dtype=np.int32)
            color = label_color_map[label]
            cv2.fillPoly(mask, [points], color)
    return mask

def text_processor(text):
    values = list(map(int, re.findall(r": (\d+)", text)))
    result_array = np.array(values)
    return result_array

def text_processor_performance(text):
    # "Score:" 뒤의 숫자를 슬래시(`/`) 여부에 관계없이 추출
    scores = re.findall(r'Score:\s*(\d+)', text)
    # 정수형 리스트로 변환하여 반환
    return [int(score) for score in scores]

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def save_text_to_file(text, file_path):
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text)
    except Exception as e:
        print(f"error while saving text.")

def main():
    path = "C:/Workspace/Accessibility/Segmentation_Finetuning/yolov11_based/Data_w_ID_2/unsupervised_test_1117/Yolo_dataset_first_cycle_1118/images/train"
    list_1 = get_files_list(path, '.png')
    print(list_1)



#main()