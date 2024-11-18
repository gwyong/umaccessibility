import time, os, glob, sys, pickle, random
import torch
from tqdm import tqdm

import utils, sam, clip

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(42)
torch.manual_seed(42)
if device == "cuda":
    torch.cuda.manual_seed(42)

# ##### Argument Preparation #####
num_input_points = 512
multimask_output = False
test_folder_path = "data/total"
output_dir = "./outputs"
query_dir = "./queries"
sam_output_path = "./outputs/sam_output.pickle"
dino_threshold = 0.3
pred_threshold = 0.7
iou_threshold  = 0.7
min_area = 50
min_ratio = 1/256
max_ratio = 0.4
top_k = 3
show = False
save = True
use_elimination = False

# text_accessibility_features = ["townhome", "apartment", "first_story", "second_story", "third_story", "fourth_story", "entrance", "stair", "ramp", "step_free", "elevator", "air_conditioning",
#                                "sidewalk", "curb_cut", "parking_lot"]
text_accessibility_features = ["crosswalk", "curb_cut", "sidewalk"]
removal_list = ["building", "pole", "traffic light", "traffic sign", "vegetation", "sky", "person", "car", "bus", "motor bike", "bike"]
################################

model_sam = sam.SAM(device=device)
model_sam.prepare_dino()

test_set = [file_name for file_name in os.listdir(test_folder_path) if file_name.endswith('.PNG') or file_name.endswith('.png')] # if file_name.endswith('.PNG') or file_name.endswith('.png') or file_name.endswith('.JPG') or file_name.endswith('.jpg')
test_set = random.sample(test_set, min(len(test_set), 100))

sam_output = {}
for file_name in tqdm(test_set, desc="Processing images..."):
    image_path = os.path.join(test_folder_path, file_name)
    image = utils.read_image_path(image_path)

    if use_elimination:
        remaining_image, removal_mask, removal_dict = model_sam.remove_irrelevant_masks(image, removal_list, dino_threshold=dino_threshold, multimask_output=multimask_output)
        image = remaining_image

    sam_output[file_name] = {}
    for accessibility_text in text_accessibility_features:
        input_boxes = model_sam.prepare_dino_boxes(image, accessibility_text, dino_threshold=dino_threshold)
        if len(input_boxes) != 0:
            inputs, outputs = model_sam.segment(image, input_boxes=[input_boxes], multimask_output=multimask_output)
            dino_masks, dino_scores = model_sam.extract_dino_masks(inputs=inputs, outputs=outputs)
            sam_output[file_name][accessibility_text] = (dino_masks, dino_scores)
    
    if use_elimination:
        sam_output[file_name] = utils.remain_relevant_masks(removal_mask, sam_output[file_name])

with open(sam_output_path, 'wb') as f:
    pickle.dump(sam_output, f)