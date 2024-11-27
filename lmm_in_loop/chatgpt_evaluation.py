import pandas as pd

from openai import OpenAI
import pandas as pd
from utils import text_processor_performance, text_processor, get_files_list, encode_image, save_text_to_file, ensure_directory_exists


api_key = "INPUT API KEY"
client = OpenAI(api_key = api_key)

def ask_image_question(image_base64):
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "The goal is to evaluate how well the image has been labeled. Sidewalks are labeled in red, curbcuts in green, and crosswalks in blue. Assess the quality of the labeling and assign a score between 1 and 10. If there are missing labels or many incorrectly labeled areas, assign a lower score. If there is no color label (e.g., for sidewalks, curbcuts, or crosswalks), evaluate whether the absence of the label is justified based on the context of the image. Provide outpout in forms of 'Score: 3/10'"
            },
            {
            "type": "image_url",
            "image_url": {
                "url":  f"data:image/jpeg;base64,{image_base64}"
            },
            },
        ],
        }
    ],
    max_tokens=20
    )

    return response.choices[0].message.content

def test_runner():
    return "adsfsdaf Score: 9."

def chat_gpt_inferencing(input_image_path, result_df_path, test_mode = False, save_raw_result = True):
    ensure_directory_exists(result_df_path + '/chatgpt_raw_result')

    columns = ["filename", "score"]
    df = pd.DataFrame(columns=columns)
    file_path = result_df_path + '/chatgpt_evaluation.csv'

    png_list = get_files_list(input_image_path, '.png')
    print("Data Number:", len(png_list))
    skip_list = []
    for i in range(len(png_list)):
        if i%100 == 0: print("Processing:", i)
        try:
            image_path = input_image_path + '/' + png_list[i]
            base64_image = encode_image(image_path)
            if test_mode == False:
                response_text = ask_image_question(base64_image)
            if test_mode == True:
                response_text = test_runner()
            if save_raw_result == True:
                text_file_path = result_df_path + '/chatgpt_raw_result/'+png_list[i][:-4]+'.txt'
                save_text_to_file(response_text, file_path = text_file_path)
            response_list = text_processor_performance(response_text)[0]
            #print(png_list[i], "Score:", response_list)
            df.loc[len(df)] = [png_list[i], response_list]
        except:
            print("Pass the result for", png_list[i])
            skip_list.append(png_list[i])
        if i%50 == 0:
            df.to_csv(file_path, index=False)
    df.to_csv(file_path, index=False)
    print("skipped_list:", skip_list)
    print("skip_length:", len(skip_list))

