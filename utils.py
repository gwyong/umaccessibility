import os, json, shutil
import pandas as pd

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
        