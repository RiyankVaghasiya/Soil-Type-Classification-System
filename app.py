# main.py
from src.components.data_ingestion import load_image_data

if __name__ == "__main__":
    df = load_image_data()
    print(df)
