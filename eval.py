import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_predictions_from_dir(output_dir):
    """
    Reads prediction files from a directory, extracts predictions and actual values, 
    and computes evaluation metrics.

    Parameters:
        output_dir: Directory containing prediction files.

    Returns:
        A dictionary of evaluation metrics (MSE, MAE).
    """
    all_predictions = []
    all_actuals = []
    
    # Iterate through all files in the directory
    for file_name in os.listdir(output_dir):
        if file_name.endswith('.txt'):  # Process only .txt files
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, 'r') as file:
                for line in file:
                    # Parse the line
                    try:
                        # Expecting the format: "Prediction: xx.xx,    Actual Value: xx.xx"
                        parts = line.split(',')
                        pred = float(parts[0].split(':')[1].strip())
                        actual = float(parts[1].split(':')[1].strip())
                        
                        all_predictions.append(pred)
                        all_actuals.append(actual)
                    except (IndexError, ValueError) as e:
                        print(f"Skipping line in {file_name}: {line.strip()} (Error: {e})")
    
    # Convert to numpy arrays for evaluation
    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)

    # Compute evaluation metrics
    mse = mean_squared_error(all_actuals, all_predictions)
    mae = mean_absolute_error(all_actuals, all_predictions)
    
    return {
        "Mean Squared Error (MSE)": mse,
        "Mean Absolute Error (MAE)": mae
    }

# Example usage
output_dir = "output_data"
metrics = evaluate_predictions_from_dir(output_dir)
print("Evaluation Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
