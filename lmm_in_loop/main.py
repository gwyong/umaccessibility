import os
from cycle import cycle

def main():
    cycle_run = cycle(4)
    cycle_run.inference_unsupervised_data()
    cycle_run.chatgpt_inference(test_mode=False)
    cycle_run.create_next_traindataset()
    cycle_run.train_yolo()
    cycle_run.evaluate_yolo_model()

main()