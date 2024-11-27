import pandas as pd
from utils import ensure_directory_exists, get_files_list, exclude_elements
import shutil

def create_next_train_dataset(file_directory, data_directory, initial_file_directory, file_name = '', filtering_score = 8):
    if file_name != '': file_path = file_directory + '/' + file_name
    else : file_path = file_directory + '/chatgpt_evaluation.csv'
    df = pd.read_csv(file_path)
    df_filtered = df[df['score'] >= filtering_score]
    print("filtered score:", len(df_filtered))

    ensure_directory_exists(data_directory + '/images')
    ensure_directory_exists(data_directory + '/labels')
    ensure_directory_exists(data_directory + '/labels/labeled')
    ensure_directory_exists(data_directory + '/labels/unlabeled')
    ensure_directory_exists(data_directory + '/images/labeled')
    ensure_directory_exists(data_directory + '/images/unlabeled')

    entire_unsupervised_list = get_files_list(initial_file_directory + '/images/unlabeled','.png')
    initial_train_list = get_files_list(initial_file_directory + '/images/labeled','.png')
    new_data_list = []
    
    for i in range(len(df_filtered)):
        name_here = df_filtered['filename'].iloc[i]
        new_data_list.append(name_here)
        image_from = initial_file_directory + '/images/unlabeled/' + name_here
        image_to = data_directory + '/images/labeled/' + name_here
        label_from = data_directory + '/inferenced/result_mask_text/' + name_here[:-4] + '.txt'
        label_to = data_directory + '/labels/labeled/' + name_here[:-4] + '.txt'
        shutil.copy(image_from, image_to)
        shutil.copy(label_from, label_to)

    new_unsupervised_list = exclude_elements(entire_unsupervised_list, new_data_list)
    print("new unsupervised list length:", len(new_unsupervised_list))
    
    for i in range(len(new_unsupervised_list)):
        name_here = new_unsupervised_list[i]
        image_from = initial_file_directory + '/images/unlabeled/' + name_here
        image_to = data_directory + '/images/unlabeled/' + name_here
        label_from = data_directory + '/inferenced/result_mask_text/' + name_here[:-4] + '.txt'
        label_to = data_directory + '/labels/unlabeled/' + name_here[:-4] + '.txt'
        shutil.copy(image_from, image_to)
        shutil.copy(label_from, label_to)

    for i in range(len(initial_train_list)):
        name_here = initial_train_list[i]
        image_from = initial_file_directory + '/images/labeled/' + name_here
        image_to = data_directory + '/images/labeled/' + name_here
        label_from = initial_file_directory + '/labels/labeled/' + name_here[:-4] + '.txt'
        label_to = data_directory + '/labels/labeled/' + name_here[:-4] + '.txt'
        shutil.copy(image_from, image_to)
        shutil.copy(label_from, label_to)    

    with open(file_directory + "/dataset_log.txt", "w") as file:
        file.write("filtered number: "+str(filtering_score) + "\n")
        file.write("new added train data: "+str(len(df_filtered)) + "\n")
        file.write("total train data: "+str(len(new_data_list)+len(initial_train_list)) + "\n")
        file.write("unsupervised data left: "+ str(len(new_unsupervised_list)))
