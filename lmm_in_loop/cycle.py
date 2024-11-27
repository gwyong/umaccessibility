import numpy as np
import os
import json
from utils import ensure_directory_exists
from inference_model import inference_unlabeled_data
from chatgpt_evaluation import chat_gpt_inferencing
from create_next_traindataset import create_next_train_dataset
from train_yolo import train_yolo_seg, fixing_yaml_file
from evaluate_model import evaluate_data, calculate_metrics

class cycle:
    yolo_dataset_base_path = "C:/Workspace/Accessibility/inloop_semisupervised_learning"
    def __init__(self, cycle_number = 0, initial_amount = 200):
        self.initial_data = initial_amount
        self.cycle_num = cycle_number
        self.data_directory = 'data_200'
        self.result_directory = 'data_200_result/cycle' + str(cycle_number)
        self.model_path = 'runs/segment'
        self.yolo_training_directory = cycle.yolo_dataset_base_path + '/' + self.data_directory

        self.train_yolo_finished = False
        self.evaluate_yolo_finished = False
        self.inference_unsupervised_data_finished = False
        self.chatgpt_inference_finished = False
        self.create_next_traindataset_finished = False

        ensure_directory_exists(self.result_directory)
    
    def load_parameter(self):
        pass

    def inference_unsupervised_data(self):
        print("="*10, "inferencing unsueprvised data", "="*10)
        model_here = self.model_path + '/yolov11l_initial' + str(self.initial_data) + '_cycle' + str(self.cycle_num-1) + '/weights/best.pt'
        inference_unlabeled_data(self.data_directory + '/initial/images/unlabeled', self.data_directory + '/cycle' + str(self.cycle_num)+'/inferenced', model_here)
        print("inferencing complete.", "\n\n")

    def chatgpt_inference(self, test_mode = False):
        print("="*10, "chatgpt evaluation start", "="*10)
        chat_gpt_inferencing(self.data_directory + '/cycle' + str(self.cycle_num) + '/inferenced/overlay', self.result_directory, test_mode = test_mode)
        print("evaluation complete.", "\n\n")

    def create_next_traindataset(self):
        print("="*10, "creating next train dataset", "="*10)
        create_next_train_dataset(self.result_directory, self.data_directory + '/cycle' + str(self.cycle_num), self.data_directory + '/initial', filtering_score= 8)
        print("dataset generation complete.")

    def train_yolo(self):
        print("="*10, "training yolo model", "="*10)
        model_name_here = 'yolov11l_initial' + str(self.initial_data) + '_cycle' + str(self.cycle_num)
        data_yaml_from = self.yolo_training_directory + '/initial/dataset.yaml'
        data_yaml_to = self.data_directory + '/cycle' + str(self.cycle_num) + '/dataset.yaml'
        fixing_yaml_file(data_yaml_from, data_yaml_to, cycle.yolo_dataset_base_path + '/' + self.data_directory, str(self.cycle_num))
        train_yolo_seg(data_yaml_to, model_name_here)

    def evaluate_yolo_model(self):
        print("="*10, "evaluating yolo model", "="*10)
        model_here = self.model_path + '/yolov11l_initial' + str(self.initial_data) + '_cycle' + str(self.cycle_num) + '/weights/best.pt'
        evaluate_data(self.data_directory + '/initial/images/test', self.result_directory, model_here)
        calculate_metrics(self.result_directory + '/result_mask', self.data_directory + '/initial/true_mask/test', self.result_directory)