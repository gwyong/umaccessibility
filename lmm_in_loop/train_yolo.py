import numpy as np
from ultralytics import YOLO
import yaml
import shutil

def fixing_yaml_file(yaml_from, yaml_to, data_path_base, cycle_num):
    shutil.copy(yaml_from, yaml_to)
    with open(yaml_to, 'r') as file:
        yaml_content = yaml.safe_load(file)
    yaml_content['path'] = data_path_base
    yaml_content['train'] = 'cycle' + str(cycle_num) + '/images/labeled'
    yaml_content['val'] = 'initial/images/test'
    new_file_path = yaml_to
    with open(new_file_path, 'w') as file:
        yaml.safe_dump(yaml_content, file)
    print(f"yaml file modified and saved: {new_file_path}")

def train_yolo_seg(data_path, save_model_name, epoch=100, imgsz=640, pretrained_model = 'pretrained_model/yolo11l-seg.pt'):
    model = YOLO(pretrained_model)
    model.train(
        #data='Yolo_dataset_first_cycle_1118/dataset.yaml',
        data = data_path,
        epochs=epoch,
        imgsz=imgsz,
        batch=4,
        name = save_model_name,
        #name='yolov11l_seg_1118_first_cycle',
        device=0,
        translate = 0.1,
        degrees = 10,
        workers=0
    )