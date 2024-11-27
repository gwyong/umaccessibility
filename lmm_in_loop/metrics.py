import numpy as np
import cv2
import json
import os

def calculate_metrics(A, B):
    # Flatten the matrices to 1D
    A_flat = A.flatten()
    B_flat = B.flatten()

    # Calculate True Positives, False Positives, and False Negatives
    TP = np.sum((A_flat == 1) & (B_flat == 1))
    FP = np.sum((A_flat == 0) & (B_flat == 1))
    FN = np.sum((A_flat == 1) & (B_flat == 0))
    TN = np.sum((A_flat == 0) & (B_flat == 0))

    # Calculate Precision, Recall, and F1 Score
    precision = TP / (TP + FP) if (TP + FP) > 0 else -1
    recall = TP / (TP + FN) if (TP + FN) > 0 else -1
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else -1

    return [TP, FN, FP, TN]

def calculate_final_from_four(matrix, printresult = True):
    return_matrix = []
    # matrix 3x4 matrix
    list_name = ['crosswalk', 'curbcut', 'sidewalk']
    for i in range(3):
        TP = matrix[i,0]
        FN = matrix[i,1]
        FP = matrix[i,2]
        TN = matrix[i,3]
        accuracy = (TP+TN)/(TP+TN+FN+FP)
        if TP+FP+FN > 0:iou = TP/(TP+FP+FN)
        else:iou = -1
        if TP+FP > 0:precision = TP/(TP+FP)
        else: precision = -1
        if TP+FN > 0:recall = TP/(TP+FN)
        else:recall = -1
        if precision + recall > 0 :f1 = 2*precision*recall/(precision+recall)
        else : f1 = -1
        if printresult == True:
            print(list_name[i], ": =====")
            print('accuracy:', round(accuracy,3), 'iou:', round(iou,3), 'precision:', round(precision,3), 'recall:', round(recall,3), 'f1:', round(f1,3))
        return_matrix.append([accuracy, iou, precision, recall, f1])
    return return_matrix

def metrics_evaluation(mask_ground_truth, mask_predicted, image_size = (640,640)):
    # pixelwise [TP, FN, FP, TN]
    pixelwise_f1_matrix = np.array([[-1,-1,-1,-1], [-1,-1,-1,-1], [-1,-1,-1,-1]], dtype = np.float64)
    # imagewise [TP, FN, FP, TN]
    imagewise_f1_matrix = np.array([[-1,-1,-1,-1], [-1,-1,-1,-1], [-1,-1,-1,-1]], dtype = np.float64)

    # crosswalk map
    crosswalk_color = [255,0,0]
    crosswalk_predicted_map = np.zeros(image_size, dtype=np.uint8)
    crosswalk_predicted_map[np.all(mask_predicted == crosswalk_color, axis=-1)] = 1
    crosswalk_ground_map = np.zeros(image_size, dtype=np.uint8)
    crosswalk_ground_map[np.all(mask_ground_truth == crosswalk_color, axis=-1)] = 1
    pixelwise_f1_matrix[0:] = calculate_metrics(crosswalk_ground_map, crosswalk_predicted_map)

    # curbcut map
    curbcut_color = [0,255,0]
    curbcut_predicted_map = np.zeros(image_size, dtype=np.uint8)
    curbcut_predicted_map[np.all(mask_predicted == curbcut_color, axis=-1)] = 1
    curbcut_ground_map = np.zeros(image_size, dtype=np.uint8)
    curbcut_ground_map[np.all(mask_ground_truth == curbcut_color, axis=-1)] = 1
    pixelwise_f1_matrix[1:] = calculate_metrics(curbcut_ground_map, curbcut_predicted_map)

    # sidewalk map
    sidewalk_color = [0,0,255]
    sidewalk_predicted_map = np.zeros(image_size, dtype=np.uint8)
    sidewalk_predicted_map[np.all(mask_predicted == sidewalk_color, axis=-1)] = 1
    sidewalk_ground_map = np.zeros(image_size, dtype=np.uint8)
    sidewalk_ground_map[np.all(mask_ground_truth == sidewalk_color, axis=-1)] = 1
    pixelwise_f1_matrix[2:] = calculate_metrics(sidewalk_ground_map, sidewalk_predicted_map)

    final_map_predicted = crosswalk_predicted_map + curbcut_predicted_map * 2 + sidewalk_predicted_map * 3
    final_map_predicted.astype(np.uint8)
    final_map_ground = crosswalk_ground_map + curbcut_ground_map * 2 + sidewalk_ground_map *3
    final_map_ground.astype(np.uint8)

    for i in range(3):
        count_sidewalk_ground = np.sum(final_map_ground == i+1)
        count_sidewalk_predicted = np.sum(final_map_predicted == i+1)
        if count_sidewalk_ground > 0 and count_sidewalk_predicted > 0: # True Positive Case
            imagewise_f1_matrix[i:] = [1,0,0,0]
        if count_sidewalk_ground > 0 and count_sidewalk_predicted == 0: # False Negative Case
            imagewise_f1_matrix[i:] = [0,1,0,0]
        if count_sidewalk_ground == 0 and count_sidewalk_predicted > 0: # False Positive Case
            imagewise_f1_matrix[i:] = [0,0,1,0]
        if count_sidewalk_ground == 0 and count_sidewalk_predicted == 0: # True Negative Case
            imagewise_f1_matrix[i:] = [0,0,0,1]

    return pixelwise_f1_matrix, imagewise_f1_matrix