import os
from train import process  

data_directory = 'stock_data/'

if __name__ == '__main__':
    for filename in os.listdir(data_directory):
        if filename.endswith('.csv'):  # Only process CSV files
            input_path = os.path.join(data_directory, filename)
            print(f"\nProcessing file: {input_path}\n")
            try:
                process(input_path)
            except ValueError as e:
                print(f"Skipping {filename} due to ValueError: {e}")