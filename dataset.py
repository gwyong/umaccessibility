import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm import trange

def collect_GSV_images(api_key, df_path, output_path="./data/GSVs"):
    df  = pd.read_csv(df_path)
    url = f"https://maps.googleapis.com/maps/api/streetview"
    headings = [(0, "north"), (90, "east"), (180, "south"), (270, "west")]
    
    for idx in trange(len(df)):
        latitude = df.loc[idx, 'latitude']
        longitude = df.loc[idx, 'longitude']
        address = df.loc[idx, 'street_names']
        
        for heading in headings:
            params = {
                'location': f'{latitude}, {longitude}',
                'size': '640x640',
                'heading': heading[0],
                'fov': 90,
                'pitch': 0,
                'key': api_key
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                img.save(os.path.join(output_path, f"{address+"_"+heading[1]}.png"))
            else:
                print(f"Error: {response.status_code} | {address} | {heading[1]}")
            time.sleep(0.001)
    return

if __name__ == "__main__":
    df_path = None
    api_key = None
    collect_GSV_images(api_key, df_path, output_path="./data/GSVs")