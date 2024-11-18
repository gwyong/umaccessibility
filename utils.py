import os, json, shutil, math, cv2
import torch
import torch.nn.functional as F

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont
from matplotlib.colors import ListedColormap

def read_image_path(path):
    return Image.open(path).convert("RGB")

def extract_image_grid_points(W, H, num_points=512):
    grid_size = math.sqrt(num_points)
    x_interval, y_interval = W/grid_size, H/grid_size
    coordinates = [(int(i*x_interval), int(j*y_interval)) for j in range(int(grid_size)) for i in range(int(grid_size))]
    
    # Adjust if the number of points is less than requested
    if len(coordinates) < num_points:
        remaining_points = num_points - len(coordinates)
        for i in range(remaining_points):
            x = int((i % int(grid_size)) * x_interval + x_interval / 2)
            y = int((i // int(grid_size)) * y_interval + y_interval / 2)
            coordinates.append((x, y))

    return coordinates[:num_points]

def wrap_points(input_points):
    return [[point] for point in input_points]

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."

def plot_results(pil_img, scores, labels, boxes):
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax), c in zip(scores, labels, boxes, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        label = f'{label}: {score:0.2f}'
        ax.text(xmin, ymin, label, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

def show_mask(mask, ax, score, color):
    cmap = ListedColormap(['none', color])
    ax.imshow(mask, cmap=cmap, alpha=0.5)
    y, x = np.where(mask > 0)
    # if len(y) > 0:
    #     ax.text(x.min(), y.min(), f'Score: {score:.3f}', color='white', fontsize=12, backgroundcolor='black')

def show_masks_on_image(raw_image, masks, scores, input_points=None):
    if len(masks.shape) == 4:
        masks = masks.squeeze(1)
    if scores.shape[0] == 1:
        scores = scores.squeeze(0)

    nb_predictions = scores.shape[0]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(np.array(raw_image))

    cmap = plt.get_cmap('hsv', nb_predictions)
    colors = [cmap(i) for i in range(nb_predictions)]

    for i in range(nb_predictions):
        mask = masks[i].cpu().detach().numpy()
        color = colors[i]
        show_mask(mask, ax, scores[i].item(), color)

    if input_points is not None:
        for point in input_points:
            x, y = point
            ax.plot(x, y, 'o', color='white', markersize=8)
            ax.text(x + 5, y, f'({x}, {y})', color='white', fontsize=10, backgroundcolor='black')

    ax.axis('off')
    plt.show()

def show_all_masks_on_image(image, mask_score_dict, alpha=0.5):
    if isinstance(image, Image.Image):
        image = np.array(image)

    num_classes = len(mask_score_dict)
    cmap = plt.get_cmap("hsv")
    colors = [cmap(i / num_classes)[:3] for i in range(num_classes)]
    colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    overlay = image_rgb.copy()

    for idx, (class_name, (dino_masks, dino_scores)) in enumerate(mask_score_dict.items()):
        color = colors[idx]

        combined_mask = torch.sum(dino_masks, dim=0).clamp(max=1) 

        mask_np = combined_mask.squeeze().cpu().numpy()
        mask_colored = np.zeros_like(image_rgb)
        mask_colored[mask_np > 0] = color

        cv2.addWeighted(mask_colored, alpha, overlay, 1 - alpha, 0, overlay)

        text_position = (10, 30 * (idx + 1))
        cv2.putText(overlay, class_name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.axis('off')
    plt.show()

def find_close_boundaries_contour(mask1, mask2, distance_threshold=20):
    # cv2.findContours uses binary image, so we need to convert the mask to binary
    _, mask1_binary = cv2.threshold(mask1.astype(np.uint8), 0.5, 255, cv2.THRESH_BINARY)
    _, mask2_binary = cv2.threshold(mask2.astype(np.uint8), 0.5, 255, cv2.THRESH_BINARY)

    contours1, _ = cv2.findContours(mask1_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(mask2_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    close_points = []
    for cnt1 in contours1:
        for p1 in cnt1:
            point = (int(p1[0][0]), int(p1[0][1]))  # convert into (x, y) format
            distances = [cv2.pointPolygonTest(cnt2, point, True) for cnt2 in contours2]
            min_distance = min(abs(d) for d in distances)
            if min_distance <= distance_threshold:
                close_points.append(point)

    close_boundary_mask = np.zeros_like(mask1)
    for point in close_points:
        close_boundary_mask[point[1], point[0]] = 255  # white color
    
    y_indices, x_indices = np.where(close_boundary_mask == 255)
    close_boundary_points = list(zip(x_indices, y_indices))
    return close_boundary_mask, close_boundary_points

def calculate_iou(mask1, mask2):
    intersection = torch.logical_and(mask1, mask2)
    union = torch.logical_or(mask1, mask2)
    if torch.sum(union).float() == 0:
        return 0.0
    iou_score = torch.sum(intersection).float() / torch.sum(union).float()
    return iou_score.item()

def calculate_dice(mask1, mask2):
    intersection = torch.logical_and(mask1, mask2).sum().item()
    sum_masks = mask1.sum().item() + mask2.sum().item()
    if sum_masks == 0:
        return 0.0
    return 2 * intersection / sum_masks

def remove_small_masks(masks, iou_scores, min_area=100, min_ratio=None, image_shape=None):
    filtered_masks = []
    filtered_scores = []
    if min_ratio is not None and image_shape is not None:
        total_area = image_shape[0] * image_shape[1]
        min_area = total_area / 256
    
    for i, (mask, score) in enumerate(zip(masks, iou_scores)):
        mask_area = torch.sum(mask).item()
        if mask_area >= min_area:
            filtered_masks.append(mask)
            filtered_scores.append(score)
    return filtered_masks, filtered_scores

def remove_large_masks(masks, iou_scores, max_ratio=0.5, image_shape=None):
    filtered_masks = []
    filtered_scores = []
    max_area = max_ratio * image_shape[0] * image_shape[1]
    for i, (mask, score) in enumerate(zip(masks, iou_scores)):
        mask_area = torch.sum(mask).item()
        if mask_area <= max_area:
            filtered_masks.append(mask)
            filtered_scores.append(score)
    return filtered_masks, filtered_scores

def remove_duplicate_masks(masks, iou_scores, threshold=0.95):
    unique_masks = []
    unique_scores = []
    for i, (mask, score) in enumerate(zip(masks, iou_scores)):
        is_duplicate = False
        for unique_mask in unique_masks:
            if calculate_iou(mask, unique_mask) >= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_masks.append(mask)
            unique_scores.append(score)
    return unique_masks, unique_scores

def remain_relevant_masks(removal_mask, sam_output_dict):
    removal_mask = removal_mask.float()  # Convert to float to match dino_masks if needed

    for target_class, (dino_masks, dino_scores) in sam_output_dict.items():
        processed_masks = []
        for mask in dino_masks:
            updated_mask = mask * (1 - removal_mask)
            updated_mask = (updated_mask > 0).float()
            processed_masks.append(updated_mask)
        processed_masks = torch.stack(processed_masks)
        
        sam_output_dict[target_class] = (processed_masks, dino_scores)
    return sam_output_dict

def save_masked_image(image, mask, output_directory, image_name, mask_number):
    image_np = np.array(image)
    white_background = np.ones_like(image_np) * 255
    
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    if mask.ndim == 4 and mask.shape[0] == 1 and mask.shape[1] == 1:
        mask = mask.squeeze(0).squeeze(0)

    masked_image = np.where(mask[..., None] == 1, image_np, white_background)

    masked_image_pil = Image.fromarray(masked_image.astype(np.uint8))

    if len(os.path.splitext(image_name)) == 1:
        image_basename = os.path.splitext(image_name)[0]
    else:
        image_basename = "".join(os.path.splitext(image_name))
    output_path = os.path.join(output_directory, image_basename)
    os.makedirs(output_path, exist_ok=True)

    output_file_path = os.path.join(output_path, f"{str(mask_number)}.png")
    masked_image_pil.save(output_file_path)

def save_bbox_masked_image(image, mask, output_directory, image_name, mask_number):
    coords = np.argwhere(mask)
    y_coords, x_coords = coords[:, 1], coords[:, 2]
    
    y_min, x_min = y_coords.min(axis=0), x_coords.min(axis=0)
    y_max, x_max = y_coords.max(axis=0), x_coords.max(axis=0)
    
    image_np = np.array(image)
    cropped_image = image_np[y_min:y_max+1, x_min:x_max+1]
    cropped_mask = mask[0, y_min:y_max+1, x_min:x_max+1]
    canvas = np.ones_like(image_np) * 255
    canvas_bbox = np.ones_like(image_np) * 255

    canvas_center_y, canvas_center_x = canvas.shape[0] // 2, canvas.shape[1] // 2
    cropped_center_y, cropped_center_x = cropped_image.shape[0] // 2, cropped_image.shape[1] // 2

    start_y = canvas_center_y - cropped_center_y
    start_x = canvas_center_x - cropped_center_x

    end_y = start_y + cropped_image.shape[0]
    end_x = start_x + cropped_image.shape[1]

    cropped_mask_expanded = np.repeat(cropped_mask[:, :, np.newaxis], 3, axis=2)
    canvas[start_y:end_y, start_x:end_x][cropped_mask_expanded] = cropped_image[cropped_mask_expanded]
    canvas_bbox[start_y:end_y, start_x:end_x] = cropped_image

    masked_image_pil = Image.fromarray(canvas.astype(np.uint8))
    bbox_image_pil = Image.fromarray(canvas_bbox.astype(np.uint8))
    image_basename = os.path.splitext(image_name)[0]
    output_path = os.path.join(output_directory, image_basename)
    os.makedirs(output_path, exist_ok=True)
    output_file_path = os.path.join(output_path, f"{str(mask_number)}.png")
    output_bbox_file_path = os.path.join(output_path, f"{str(mask_number)}_bbox.png")
    masked_image_pil.save(output_file_path)
    bbox_image_pil.save(output_bbox_file_path)

def check_word_in_text(word, text):
    if word.lower() in text.lower():
        return True
    return False

def json_to_masks(json_path):
    with open(json_path, 'r') as file:
        annotation = json.load(file)
    
    mask_dict = {}
    for shape in annotation['shapes']:
        if shape["shape_type"] != "polygon":
            continue
        label = shape['label']
        points = [(int(point[1]), int(point[0])) for point in shape['points']]
        
        mask = Image.new('L', (annotation['imageHeight'], annotation['imageWidth']), 0)
        ImageDraw.Draw(mask).polygon(points, outline=1, fill=1)
        mask = np.array(mask).transpose(1, 0)

        if label not in mask_dict:
            mask_dict[label] = []
        mask_dict[label].append(mask)

    for label, masks in mask_dict.items():
        masks_array = np.stack(masks).reshape(len(masks), 1, annotation['imageHeight'], annotation['imageWidth'])
        mask_dict[label] = masks_array
        
    return annotation['imagePath'], mask_dict

def apply_mask(image, mask, color, alpha=0.5):
    image = image.convert("RGBA")
    mask = mask.convert("L")
    assert image.size == mask.size

    mask_data = np.array(mask)*255
    rgba_mask = np.zeros((*mask_data.shape, 4), dtype=np.uint8)
    
    rgba_mask[..., :3] = color
    rgba_mask[..., 3] = mask_data * alpha
    
    mask_image = Image.fromarray(rgba_mask)
    image = Image.alpha_composite(image, mask_image)
    return image

def add_label(image, label, position, color=(255, 255, 255), font_size=20):
    draw = ImageDraw.Draw(image)
    # font = ImageFont.load_default()
    font = ImageFont.truetype("arial.ttf", font_size)
    draw.text(position, label, fill=color, font=font)
    return image

def identify_nearest_query(query_dict, embedding_dict, top_k=3):
        query_dict = {k: F.normalize(v, p=2, dim=-1) for k, v in query_dict.items()}
        embedding_dict = {k: F.normalize(v, p=2, dim=-1) for k, v in embedding_dict.items()}
        results = {}

        for query_name, query_embeddings in query_dict.items():
            distances = []

            for embed_name, embedding in embedding_dict.items():
                cos_sim = torch.matmul(query_embeddings, embedding.T).squeeze()  # (# of queries,)
                distance_sum = cos_sim.sum().item()  # Sum of distances over all queries
                distances.append((embed_name, distance_sum))

            distances = sorted(distances, key=lambda x: x[1], reverse=True)[:top_k]
            
            results[query_name] = distances
        return results

def transfer_images_and_jsons(source_dir, target_dir, consider_empty_annotation=False):
    """
    Input:
    source_dir: str, directory where the images and jsons are stored
    target_dir: str, directory where the images and jsons will be copied
    consider_empty_annotation: bool, whether to consider the images with empty annotations or not
    """

    df_elizabeth = pd.read_csv("nj_intersections_tl_2020_elizabeth_sample_300.csv")
    df_southJersey = pd.read_csv("nj_intersections_tl_2020_south_jersey_sample_300.csv")

    for df_area in [df_elizabeth, df_southJersey]:
        if df_area.equals(df_elizabeth):
            area_name = "elizabeth"
        else:
            area_name = "south_jersey"
        xyid_list = df_area['xyid'].tolist()
        
        aerial_foldername = "aerial"
        street_foldername = "street"

        source_street_dir = os.path.join(source_dir, street_foldername, area_name)
        
        image_files = [f for f in os.listdir(source_street_dir) if f.endswith('.jpg') or f.endswith('.png')]
        json_files = [f for f in os.listdir(source_street_dir) if f.endswith('.json')]
        
        count = 0
        num_masks = 0

        ## transfer street images and jsons
        for image_file in image_files:
            json_file = image_file.replace('.jpg', '.json').replace('.png', '.json')
            json_file_path = os.path.join(source_street_dir, json_file)
            image_file_path = os.path.join(source_street_dir, image_file)

            if not os.path.exists(json_file_path) and not consider_empty_annotation:
                continue
            elif not os.path.exists(json_file_path):
                image_dst_path = os.path.join(target_dir, "_".join([area_name, street_foldername, image_file]))
                shutil.copy(image_file_path, image_dst_path)
                count += 1
                continue
            
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            
            if not data.get('shapes', []) and not consider_empty_annotation:
                continue

            image_dst_path = os.path.join(target_dir, "_".join([area_name, street_foldername, image_file]))
            json_dst_path = os.path.join(target_dir, "_".join([area_name, street_foldername, json_file]))
            shutil.copy(image_file_path, image_dst_path)
            shutil.copy(json_file_path, json_dst_path)

            count += 1
            num_masks += len(data['shapes'])
        print(f"Total {count} images copied from {area_name} | {street_foldername} and number of masks: {num_masks}")

        ## transfer aerial images and jsons
        count = 0
        num_masks = 0
        for xyid in os.listdir(os.path.join(source_dir, aerial_foldername, area_name)):
            source_aerial_dir = os.path.join(source_dir, aerial_foldername, area_name, xyid)
            image_files = [f for f in os.listdir(source_aerial_dir) if f.endswith('.jpg') or f.endswith('.png')]
            json_files = [f for f in os.listdir(source_aerial_dir) if f.endswith('.json')]
            if json_files == [] and not consider_empty_annotation:
                continue
            elif json_files == []:
                image_file_path = os.path.join(source_aerial_dir, image_files[0])
                image_dst_path = os.path.join(target_dir, "_".join([area_name, aerial_foldername, xyid, image_files[0]]))
                shutil.copy(image_file_path, image_dst_path)
                count += 1
                continue
            else:
                with open(os.path.join(source_aerial_dir, json_files[0]), 'r') as f:
                    data = json.load(f)
                if not data.get('shapes', []) and not consider_empty_annotation:
                    continue
                
                image_file_path = os.path.join(source_aerial_dir, image_files[0])
                json_file_path = os.path.join(source_aerial_dir, json_files[0])

                image_dst_path = os.path.join(target_dir, "_".join([area_name, aerial_foldername, xyid, image_files[0]]))
                json_dst_path = os.path.join(target_dir, "_".join([area_name, aerial_foldername, xyid, json_files[0]]))
                
                shutil.copy(image_file_path, image_dst_path)
                shutil.copy(json_file_path, json_dst_path)
                count += 1
                num_masks += len(data['shapes'])
        print(f"Total {count} images copied from {area_name} | {aerial_foldername} and number of masks: {num_masks}")

if __name__ == "__main__":
    source_dir = "data"
    target_dir = "data/total"
    transfer_images_and_jsons(source_dir, target_dir, consider_empty_annotation=True)
        