1. Use main.py file to run the code.
2. cycle.py contains main functions for training.

2.1. inference_unsupervised_data
- using model from pervious cycle, inference all unlabeled data and create a segmentation mask
- this mask may not very accurate

2.2. chatgpt_inference
- this use chatgpt to evaluate the segmentation result
- need to input your own ChatGPT api key on the line 8 in chatgpt_evaluation.py
- Running this part cost you a money
- Inference result will be saved as a csv file

2.3. create_next_dataset
- based on the chatgpt evaluation, this code filter the files and generate new yolo dataset.
- It will automatically create a training folder and copy the file
- You may need to check dataset.yaml file for directory on your own computer

2.4. train_yolo
- based on the directory prepared in 2.3. this part run the yolo training code.
- You may need to check dataset.yaml file for directory on your own computer

2.5. evaluate_yolo_model
- this part automatically evaluate the new yolo model created in 2.4.
- the result will be saved as a csv dataframe file.