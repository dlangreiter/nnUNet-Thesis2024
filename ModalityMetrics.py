# performance_metrics_calculator.py

import os
import sys
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import re
from collections import defaultdict

# Define the modality mappings based on exact filenames
BRAT_MODALITIES = {
    'Liver_0004_0000.nii.gz': 'Axial Precontrast Fat Suppressed T1w (dynpre)',
    'Liver_0004_0001.nii.gz': 'Axial Opposede Phase (opposed)',
    'Liver_0004_0002.nii.gz': 'Axial In Phase (t1nfs)',
    'Liver_0005_0000.nii.gz': 'Axial Precontrast Fat Suppressed T1w (dynpre)',
    'Liver_0005_0001.nii.gz': 'Axial Opposede Phase (opposed)',
    'Liver_0005_0002.nii.gz': 'Axial In Phase (t1nfs)',
    'Liver_0010_0000.nii.gz': 'Axial Precontrast Fat Suppressed T1w (dynpre)',
    'Liver_0010_0001.nii.gz': 'Axial Opposede Phase (opposed)',
    'Liver_0010_0002.nii.gz': 'Axial In Phase (t1nfs)',
    'Liver_0028_0000.nii.gz': 'Axial Precontrast Fat Suppressed T1w (dynpre)',
    'Liver_0028_0001.nii.gz': 'Axial Opposede Phase (opposed)',
    'Liver_0028_0002.nii.gz': 'Axial In Phase (t1nfs)',
    'Liver_0031_0000.nii.gz': 'Axial Precontrast Fat Suppressed T1w (dynpre)',
    'Liver_0031_0001.nii.gz': 'Axial Opposede Phase (opposed)',
    'Liver_0031_0002.nii.gz': 'Axial In Phase (t1nfs)',
    'Liver_0037_0000.nii.gz': 'Axial Precontrast Fat Suppressed T1w (dynpre)',
    'Liver_0037_0001.nii.gz': 'Axial Opposede Phase (opposed)',
    'Liver_0037_0002.nii.gz': 'Axial In Phase (t1nfs)',
    'Liver_0040_0000.nii.gz': 'Axial Precontrast Fat Suppressed T1w (dynpre)',
    'Liver_0040_0001.nii.gz': 'Axial Opposede Phase (opposed)',
    'Liver_0040_0002.nii.gz': 'Axial In Phase (t1nfs)',
    'Liver_0050_0000.nii.gz': 'Axial Precontrast Fat Suppressed T1w (dynpre)',
    'Liver_0050_0001.nii.gz': 'Axial Opposede Phase (opposed)',
    'Liver_0050_0002.nii.gz': 'Axial In Phase (t1nfs)'
    # Add all relevant filename-to-modality mappings here
}

def get_folder_path(prompt):
    """
    Prompts the user to input a folder path and validates its existence.

    Args:
        prompt (str): The prompt message to display to the user.

    Returns:
        str: Absolute path to the folder.
    """
    while True:
        path = input(prompt).strip('"').strip("'")
        if not os.path.isdir(path):
            print(f"Error: The directory '{path}' does not exist or is not accessible. Please try again.\n")
        else:
            return os.path.abspath(path)

def calculate_dice_coefficient(y_true, y_pred):
    """
    Calculates the DICE coefficient between two binary masks.

    Args:
        y_true (numpy.ndarray): Ground truth binary mask.
        y_pred (numpy.ndarray): Predicted binary mask.

    Returns:
        float: DICE coefficient.
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    if np.sum(y_true_f) == 0 and np.sum(y_pred_f) == 0:
        return 1.0  # Both masks are empty
    intersection = np.sum(y_true_f * y_pred_f)
    dice = (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f))
    return dice

def calculate_iou(y_true, y_pred):
    """
    Calculates the Intersection over Union (IoU) between two binary masks.

    Args:
        y_true (numpy.ndarray): Ground truth binary mask.
        y_pred (numpy.ndarray): Predicted binary mask.

    Returns:
        float: IoU score.
    """
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0:
        return 1.0  # Both masks are empty
    iou = intersection / union
    return iou

def calculate_precision(y_true, y_pred):
    """
    Calculates the Precision between two binary masks.

    Args:
        y_true (numpy.ndarray): Ground truth binary mask.
        y_pred (numpy.ndarray): Predicted binary mask.

    Returns:
        float: Precision score.
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    if np.sum(y_pred_f) == 0:
        return np.nan  # Undefined Precision
    precision = precision_score(y_true_f, y_pred_f, zero_division=0)
    return precision

def calculate_recall(y_true, y_pred):
    """
    Calculates the Recall between two binary masks.

    Args:
        y_true (numpy.ndarray): Ground truth binary mask.
        y_pred (numpy.ndarray): Predicted binary mask.

    Returns:
        float: Recall score.
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    if np.sum(y_true_f) == 0:
        return np.nan  # Undefined Recall
    recall = recall_score(y_true_f, y_pred_f, zero_division=0)
    return recall

def calculate_auroc(y_true, y_scores):
    """
    Calculates the AUROC between the ground truth and prediction scores.

    Args:
        y_true (numpy.ndarray): Ground truth binary mask.
        y_scores (numpy.ndarray): Predicted scores or probabilities.

    Returns:
        float: AUROC score.
    """
    try:
        # Flatten the arrays
        y_true_f = y_true.flatten()
        y_scores_f = y_scores.flatten()
        # Check if there are both classes present
        if len(np.unique(y_true_f)) < 2:
            return np.nan  # AUROC is not defined in this case
        auroc = roc_auc_score(y_true_f, y_scores_f)
        return auroc
    except Exception as e:
        print(f"Error calculating AUROC: {e}")
        return np.nan

def load_nifti_file(filepath):
    """
    Loads a NIfTI file and returns its data as a numpy array.

    Args:
        filepath (str): Path to the NIfTI file.

    Returns:
        numpy.ndarray: Image data.
    """
    try:
        img = nib.load(filepath)
        data = img.get_fdata()
        return data
    except Exception as e:
        print(f"Error loading NIfTI file '{filepath}': {e}")
        return None

def main():
    print("=== Performance Metrics Calculator for MRI Images ===\n")
    
    # Get folder paths from user
    pred_folder = get_folder_path("Enter the path to the predicted NIfTI files folder: ")
    gt_folder = get_folder_path("Enter the path to the ground truth NIfTI files folder: ")
    
    # Initialize lists to store results
    results = []
    
    # List all NIfTI files in the predicted folder
    pred_files = [f for f in os.listdir(pred_folder) if f.lower().endswith(('.nii', '.nii.gz'))]
    
    if not pred_files:
        print(f"No NIfTI files found in the predicted folder: {pred_folder}")
        sys.exit(1)
    
    print(f"\nFound {len(pred_files)} predicted NIfTI files.\n")
    
    # Iterate over predicted files
    for pred_file in pred_files:
        # Determine modality based on filename
        modality = BRAT_MODALITIES.get(pred_file, 'Unknown')
        
        # Construct full file paths
        pred_path = os.path.join(pred_folder, pred_file)
        gt_file = pred_file  # Assuming ground truth file has the same name
        gt_path = os.path.join(gt_folder, gt_file)
        
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth file '{gt_file}' not found for prediction '{pred_file}'. Skipping.")
            continue
        
        # Load NIfTI files
        pred_data = load_nifti_file(pred_path)
        gt_data = load_nifti_file(gt_path)
        
        if pred_data is None or gt_data is None:
            print(f"Warning: Failed to load data for '{pred_file}'. Skipping.")
            continue
        
        # Ensure the shapes match
        if pred_data.shape != gt_data.shape:
            print(f"Warning: Shape mismatch for '{pred_file}'. Predicted shape: {pred_data.shape}, Ground truth shape: {gt_data.shape}. Skipping.")
            continue
        
        # Binarize the ground truth and prediction
        gt_binary = (gt_data > 0).astype(np.uint8)
        # If prediction is probabilistic, keep as is; if binary, binarize
        if pred_data.dtype != float and pred_data.dtype != np.float32 and pred_data.dtype != np.float64:
            pred_binary = (pred_data > 0).astype(np.uint8)
        else:
            pred_binary = pred_data  # Assume it's probabilistic
        
        # For metrics that require binary predictions
        if pred_data.dtype != float and pred_data.dtype != np.float32 and pred_data.dtype != np.float64:
            pred_binary_for_metrics = pred_binary
        else:
            # Threshold the probabilistic predictions at 0.5 for binary metrics
            pred_binary_for_metrics = (pred_data > 0.5).astype(np.uint8)
        
        # Calculate DICE coefficient
        dice = calculate_dice_coefficient(gt_binary, pred_binary_for_metrics)
        
        # Calculate IoU
        iou = calculate_iou(gt_binary, pred_binary_for_metrics)
        
        # Calculate Precision
        precision = calculate_precision(gt_binary, pred_binary_for_metrics)
        
        # Calculate Recall
        recall = calculate_recall(gt_binary, pred_binary_for_metrics)
        
        # Calculate AUROC
        # For binary masks, flatten and use predicted probabilities or binary predictions
        if pred_data.dtype in [float, np.float32, np.float64]:
            y_scores = pred_data.flatten()
        else:
            y_scores = pred_binary.flatten()
        auroc = calculate_auroc(gt_binary, y_scores)
        
        # Append results
        results.append({
            'Filename': pred_file,
            'Modality': modality,
            'DICE': dice,
            'IOU': iou,
            'Precision': precision,
            'Recall': recall,
            'AUROC': auroc
        })
    
    if not results:
        print("No valid prediction-ground truth pairs found. Exiting.")
        sys.exit(1)
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Group by Modality and calculate mean and standard deviation for all metrics
    summary = df.groupby('Modality').agg({
        'DICE': ['mean', 'std'],
        'IOU': ['mean', 'std'],
        'Precision': ['mean', 'std'],
        'Recall': ['mean', 'std'],
        'AUROC': ['mean', 'std']
    }).reset_index()
    
    # Flatten MultiIndex columns
    summary.columns = ['Modality', 
                       'DICE_Mean', 'DICE_SD',
                       'IOU_Mean', 'IOU_SD',
                       'Precision_Mean', 'Precision_SD',
                       'Recall_Mean', 'Recall_SD',
                       'AUROC_Mean', 'AUROC_SD']
    
    # Handle NaN values in standard deviations (if any modality has only one sample, std will be NaN)
    metrics = ['DICE_SD', 'IOU_SD', 'Precision_SD', 'Recall_SD', 'AUROC_SD']
    for metric in metrics:
        summary[metric] = summary[metric].fillna('Undefined')
    
    # Display the summary
    print("\n=== Summary of Performance Metrics by Modality ===\n")
    print(summary.to_string(index=False))
    
    # **New Addition: Print Images Included in Each Modality**
    print("\n=== Images Included in Each Modality ===\n")
    # Create a dictionary mapping modalities to list of Filenames
    modality_to_files = df.groupby('Modality')['Filename'].apply(list).to_dict()
    for modality, files in modality_to_files.items():
        print(f"Modality: {modality}")
        print(f"Number of Images: {len(files)}")
        print(f"Filenames: {', '.join(files)}\n")
    
    # Export to Excel
    output_path = os.path.join(os.getcwd(), 'performance_metrics_summary.xlsx')
    try:
        with pd.ExcelWriter(output_path) as writer:
            # Write summary to the first sheet
            summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Write detailed results to a second sheet
            df.to_excel(writer, sheet_name='Detailed_Results', index=False)
        
        print(f"\nSummary successfully saved to '{output_path}'")
    except Exception as e:
        print(f"Error saving summary to Excel: {e}")
    
    print("\n=== Performance Metrics Calculation Completed ===")

if __name__ == "__main__":
    main()
