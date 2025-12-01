from datasets import load_dataset
import pandas as pd
import os

def download_and_save_data():
    print("Downloading dataset 'Pulk17/Fake-News-Detection'...")
    try:
        dataset_name = "Pulk17/Fake-News-Detection-dataset"
        print(f"Downloading dataset '{dataset_name}'...")
        dataset = load_dataset(dataset_name)
        print("Dataset downloaded successfully.")
        
        # Check dataset structure
        print(dataset)
        
        # Convert to pandas DataFrame
        # Usually huggingface datasets have 'train', 'test', 'validation' splits
        # We will combine them or just save the available ones.
        
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        for split in dataset.keys():
            df = dataset[split].to_pandas()
            output_file = os.path.join(data_dir, f"fake_news_{split}.csv")
            df.to_csv(output_file, index=False)
            print(f"Saved {split} split to {output_file} with {len(df)} records.")
            
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    download_and_save_data()

