import os
from train import process  

input_dir = 'stock_data/'
output_dir = 'output_data/'

if __name__ == '__main__':
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):  # Only process CSV files
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            predict_path = os.path.join(output_dir, filename + '.txt')
            print(f"\nProcessing file: {input_path}\n")
            try:
                process(input_path, output_path, predict_path)
            except ValueError as e:
                print(f"Skipping {filename} due to ValueError: {e}")