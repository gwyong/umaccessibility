import cv2
import numpy as np
from ultralytics import YOLO
from utils import get_files_list, ensure_directory_exists
from metrics import *
import pandas as pd

color_list = [[255,0,0], [0,255,0], [0,0,255]]

def processing_one_image(img, model, color_list, confidence_threshold = 0.25):
    img = cv2.resize(img, (640, 640))
    mask_total = np.zeros((640,640,3))
    results = model(img,verbose=False,conf=confidence_threshold,retina_masks=True, device = 'cuda')

    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    confidences = results[0].boxes.conf.tolist()
    masks = results[0].masks
    if results[0].masks is not None:
        for i in range(len(classes)):
            if confidences[i] > confidence_threshold:
                clss = results[0].boxes.cls.cpu().tolist()
                class_here = int(classes[i])
                masks = results[0].masks.xy
                coordinates_mask = masks[i]
                contour = coordinates_mask.reshape((-1, 1, 2)).astype(np.int32)
                mask_total = cv2.fillPoly(mask_total, [contour], color=color_list[class_here])
    added = cv2.addWeighted(img.astype(np.uint8), 0.5, mask_total.astype(np.uint8), 0.5, 0)
    return added, mask_total

def evaluate_data(input_folder, output_folder_lower_level, model_path):
    try:
        model = YOLO(model_path)
        ensure_directory_exists(output_folder_lower_level + '/overlay')
        ensure_directory_exists(output_folder_lower_level + '/result_mask')
        file_path_list = get_files_list(input_folder, '.png')
        print("Data amount:", len(file_path_list))
    except:
        print("error on preparation stage.")
        exit()

    for i in range(len(file_path_list)):
        try:
            img_name = file_path_list[i]
            img_load_path = input_folder + '/' + img_name
            img = cv2.imread(img_load_path)        
            predicted_added, mask_predicted = processing_one_image(img, model, color_list)
            
            compared_image = predicted_added
            cv2.imshow('compared_image', compared_image)
            cv2.waitKey(1)
            compared_image = cv2.resize(compared_image, (400, 400))
            save_img_name = output_folder_lower_level + '/overlay/' + img_name
            cv2.imwrite(save_img_name, compared_image)
            save_img_name = output_folder_lower_level + '/result_mask/' + img_name
            cv2.imwrite(save_img_name, mask_predicted)
        except:
            print("error on image ", file_path_list[i])

def calculate_metrics(predicted_mask_folder, ground_truth_mask_folder, result_path):
    predicted_mask_list = get_files_list(predicted_mask_folder, '.png')
    imagewise_metrics = np.zeros((len(predicted_mask_list), 3, 4)) #(Data_N, Class_N, Counter)
    pixelwise_metrics_mean = [[[],[],[],[],[]],
                              [[],[],[],[],[]],
                              [[],[],[],[],[]],
                              [[],[],[],[],[]]]    # ([accuracy, iou, precision, recall, f1])
    
    print('predicted mask list:', len(predicted_mask_list))
    for i in range(len(predicted_mask_list)):
        predicted_mask = cv2.imread(predicted_mask_folder + '/' + predicted_mask_list[i])
        ground_truth_mask = cv2.imread(ground_truth_mask_folder + '/' + predicted_mask_list[i])
        pixelwise_f1, imagewise_f1 = metrics_evaluation(ground_truth_mask, predicted_mask, image_size = (640,640))
        imagewise_metrics[i,:,:] = imagewise_f1

        f1_calculation_matrix = calculate_final_from_four(pixelwise_f1, printresult=False)
        for j in range(3):
            for k in range(5):
                if f1_calculation_matrix[j][k] != -1:
                    pixelwise_metrics_mean[j][k].append(f1_calculation_matrix[j][k])
    
    sum_imagewise = np.sum(imagewise_metrics, axis = 0)
    print("="*100)
    print("image wise metric:")
    print(sum_imagewise)
    return_imagewise_array = np.zeros((4,5))
    pixelwise_metric_final = calculate_final_from_four(sum_imagewise)
    list_name = ['crosswalk', 'curbcut', 'sidewalk', 'average']
    metric_name = ['accuracy', 'iou', 'precision', 'recall', 'f1']
    return_imagewise_array[:3,:] = pixelwise_metric_final
    return_imagewise_array[3,:] = np.mean(pixelwise_metric_final, axis = 0)
    df = pd.DataFrame(return_imagewise_array, index=list_name, columns=metric_name)
    csv_file_path = result_path + '/imagewise_performance.csv'
    df.to_csv(csv_file_path, index=True)

    print("="*100)
    print("pixel wise metric averaging by image:")
    list_name = ['crosswalk', 'curbcut', 'sidewalk', 'average']
    metric_name = ['accuracy', 'iou', 'precision', 'recall', 'f1']

    return_numpy_array = np.zeros((4,5))

    for i in range(3):
        print(list_name[i], ': =====')
        str_print = ''
        for j in range(5):
            str_print += metric_name[j] + ': ' + str(round(np.mean(pixelwise_metrics_mean[i][j]), 3)) + '('+str(len(pixelwise_metrics_mean[i][j]))+') '
            return_numpy_array[i,j] = np.mean(pixelwise_metrics_mean[i][j])
        print(str_print)    
    
    for i in range(5):
        return_numpy_array[-1,i] = np.mean(return_numpy_array[:-1,i])
    df = pd.DataFrame(return_numpy_array, index=list_name, columns=metric_name)
    csv_file_path = result_path + '/pixelwise_performance.csv'
    df.to_csv(csv_file_path, index=True)