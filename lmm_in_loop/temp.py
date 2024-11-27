import os
from utils import get_files_list, exclude_elements
import shutil

def generate_cycle1_data():
    path_A = "C:/Workspace/Accessibility/Segmentation_Finetuning/yolov11_based/Data_w_ID_2/unsupervised_test_1117/Yolo_dataset_first_cycle_1118/images/train"
    path_B = "C:/Workspace/Accessibility/Segmentation_Finetuning/yolov11_based/Data_w_ID_2/Yolo_Dataset_street_100_split/images/unsupervised"
    path_B_label = "C:/Workspace/Accessibility/Segmentation_Finetuning/yolov11_based/Data_w_ID_2/Yolo_Dataset_street_100_split/labels/unsupervised"
    path_to = "C:/Workspace/Accessibility/Segmentation_Finetuning/yolov11_based/Data_w_ID_2/unsupervised_test_1117/Yolo_dataset_first_cycle_1118"
    new_train_data_list = get_files_list(path_A, '.png')
    old_unsupervised_data_list = get_files_list(path_B, '.png')
    new_unsupervised_data_list = exclude_elements(old_unsupervised_data_list, new_train_data_list)

    print("len new train data list:", len(new_train_data_list))
    print("len old supervised data list:", len(old_unsupervised_data_list))
    print("len new unsupervised data list:", len(new_unsupervised_data_list))

    for i in range(len(new_unsupervised_data_list)):
        png_name = new_unsupervised_data_list[i]
        img_path_from = path_B + '/' + png_name
        img_path_to = path_to + '/images/unsupervised/' + png_name
        img_label_from = "C:/Workspace/Accessibility/Segmentation_Finetuning/yolov11_based/Data_w_ID_2/unsupervised_test_1117/result_mask_yolo" + '/' + png_name[:-4] + '.txt'
        img_label_to = path_to + '/labels/unsupervised/' + png_name[:-4] + '.txt'
        try:
            shutil.copy(img_path_from, img_path_to)
            shutil.copy(img_label_from, img_label_to)
        except:
            print("error on img:", png_name)

generate_cycle1_data()
