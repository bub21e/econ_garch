import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_predictions_per_file(output_dir):
    """
    Reads prediction files from a directory, computes evaluation metrics 
    (MSE and MAE) for each file, and prints the results.

    Parameters:
        output_dir: Directory containing prediction files.
    """
    # Iterate through all files in the directory
    for file_name in os.listdir(output_dir):
        if file_name.endswith('.txt'):  # Process only .txt files
            file_path = os.path.join(output_dir, file_name)
            predictions = []
            actuals = []
            
            # Read predictions and actual values from the file
            with open(file_path, 'r') as file:
                for line in file:
                    try:
                        # Expecting the format: "Prediction: xx.xx,    Actual Value: xx.xx"
                        parts = line.split(',')
                        pred = float(parts[0].split(':')[1].strip())
                        actual = float(parts[1].split(':')[1].strip())
                        
                        predictions.append(pred)
                        actuals.append(actual)
                    except (IndexError, ValueError) as e:
                        print(f"Skipping line in {file_name}: {line.strip()} (Error: {e})")
            
            # Convert to numpy arrays for evaluation
            predictions = np.array(predictions)
            actuals = np.array(actuals)

            # Compute evaluation metrics
            mse = mean_squared_error(actuals, predictions) if len(predictions) > 0 else float('nan')
            mae = mean_absolute_error(actuals, predictions) if len(predictions) > 0 else float('nan')

            # Print metrics for the file
            print(f"Metrics for {file_name}:")
            print(f"  Mean Squared Error (MSE): {mse:.4f}")
            print(f"  Mean Absolute Error (MAE): {mae:.4f}")
            print("-" * 40)

# Example usage
output_dir = "output_data"  # Replace with your actual directory path
evaluate_predictions_per_file(output_dir)
