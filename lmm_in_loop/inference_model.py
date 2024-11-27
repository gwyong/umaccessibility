import cv2
import numpy as np
from ultralytics import YOLO
from utils import get_files_list, ensure_directory_exists

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

def mask_to_yolo_labels(mask_image, output_path):
    mask = mask_image
    color_to_class = {
        (255, 0, 0): 0,  # Red -> Class 0
        (0, 255, 0): 1,  # Green -> Class 1
        (0, 0, 255): 2   # Blue -> Class 2
    }
    # Get image dimensions
    height, width, _ = mask.shape
    # Prepare YOLO label list
    yolo_labels = []
    for color, class_id in color_to_class.items():
        # Create a binary mask for the current color
        binary_mask = cv2.inRange(mask, np.array(color), np.array(color))
        # Find contours for the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            # Normalize segmentation points
            normalized_points = [
                (point[0][0] / width, point[0][1] / height) for point in contour
            ]
            # Flatten points into a single list
            segmentation = [coord for point in normalized_points for coord in point]
            # Filter out small noise
            if len(segmentation) >= 6:  # Minimum 3 points (x, y)
                line = f"{class_id} " + " ".join(f"{coord:.6f}" for coord in segmentation)
                yolo_labels.append(line)
    with open(output_path, 'w') as file:
        file.write("\n".join(yolo_labels))

def inference_unlabeled_data(input_folder, output_folder_lower_level, model_path):
    try:
        model = YOLO(model_path)
        ensure_directory_exists(output_folder_lower_level + '/overlay')
        ensure_directory_exists(output_folder_lower_level + '/result_mask')
        ensure_directory_exists(output_folder_lower_level + '/result_mask_text')

        file_path_list = get_files_list(input_folder, '.png')
        print("Data amount:", len(file_path_list))
    except:
        print("error on preparation stage.")
        exit()

    for i in range(len(file_path_list)):
        #try:
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
        save_yolo_label_path = output_folder_lower_level + '/result_mask_text/' + img_name[:-4] + '.txt'
        mask_to_yolo_labels(mask_predicted, save_yolo_label_path)
        #except:
        #    print("error on image ", file_path_list[i])