# src/data_ingestion.py
import os
import pandas as pd

def load_image_data(base_path='./data/raw/soil-types'):
    data = []
    for label in os.listdir(base_path):
        label_path = os.path.join(base_path, label)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                if file.endswith((".jpg", ".jpeg", ".png")):
                    data.append({
                        "image_path": os.path.join(label_path, file),
                        "label": label
                    })
    df = pd.DataFrame(data)
    return df
