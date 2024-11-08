import os
import shutil
import SimpleITK as sitk
import numpy as np
import json
import pandas as pd
from glob import glob
from collections import Counter

# -----------------------------
# Configuration Parameters
# -----------------------------

# Path to the dataset's parent directory
DATASET_DIR = '/home/declan/MultiModalModel/0001(95)'  # Update if different

# Path to the nnU-Net raw dataset directory
NNUNET_RAW_DIR = '/home/declan/MultiModalModel/nnUNet_raw'  # Update if different

# Unique Dataset ID and Name
DATASET_ID = 100  # As per your output, assuming Dataset100
DATASET_NAME = 'Liver'

# File format for nnU-Net
FILE_ENDING = '.nii.gz'

# Path to the CSV file mapping PatientID to DLDS
DLDS_MAPPING_CSV = '/home/declan/MultiModalModel/patient_dlss.csv'  # Update if different

# Series to Label mapping with DLDS as part of the key
SERIES_TO_LABEL = {
    (1, '16'): 'H',
    (1, '17'): 'G',
    (1, '32'): 'B',
    (1, '34'): 'C',
    (2, '7'): 'H',
    (2, '8'): 'G',
    (2, '11'): 'B',
    (3, '3'): 'G',
    (3, '4'): 'H',
    (3, '10'): 'B',
    (3, '14'): 'K',
    (4, '3'): 'G',
    (4, '4'): 'H',
    (4, '5'): 'B',
    (4, '19'): 'K',
    (5, '3'): 'B',
    (5, '4'): 'H',
    (5, '5'): 'G',
    (5, '11'): 'K',
    (6, '5'): 'H',
    (6, '11'): 'B',
    (6, '15'): 'K',
    (7, '4'): 'B',
    (7, '5'): 'H',
    (7, '6'): 'G',
    (7, '8'): 'B',
    (7, '13'): 'K',
    (8, '6'): 'B',
    (8, '603'): 'G',
    (8, '604'): 'H',
    (9, '4'): 'B',
    (9, '5'): 'H',
    (9, '6'): 'G',
    (10, '3'): 'G',
    (10, '4'): 'H',
    (10, '5'): 'B',
    (10, '23'): 'K',
    (11, '4'): 'G',
    (11, '5'): 'H',
    (11, '26'): 'B',
    (11, '30'): 'K',
    (12, '3'): 'G',
    (12, '4'): 'H',
    (12, '10'): 'B',
    (12, '14'): 'K',
    (13, '5'): 'B',
    (13, '7'): 'G',
    (13, '13'): 'K',
    (14, '6'): 'B',
    (14, '7'): 'H',
    (14, '8'): 'G',
    (14, '15'): 'K',
    (15, '4'): 'B',
    (15, '5'): 'H',
    (15, '6'): 'G',
    (15, '19'): 'K',
    (16, '3'): 'G',
    (16, '4'): 'H',
    (16, '16'): 'B',
    (16, '21'): 'K',
    (17, '13'): 'H',
    (17, '20'): 'B',
    (17, '25'): 'K',
    (18, '3'): 'G',
    (18, '4'): 'H',
    (18, '12'): 'B',
    (18, '16'): 'K',
    (19, '4'): 'G',
    (19, '5'): 'H',
    (19, '13'): 'B',
    (19, '17'): 'K',
    (20, '3'): 'G',
    (20, '4'): 'H',
    (20, '12'): 'B',
    (20, '16'): 'K',
    (21, '3'): 'G',
    (21, '4'): 'H',
    (21, '7'): 'B',
    (21, '13'): 'K',
    (22, '10'): 'B',
    (23, '4'): 'B',
    (23, '5'): 'H',
    (23, '6'): 'G',
    (24, '3'): 'G',
    (24, '4'): 'H',
    (24, '12'): 'B',
    (24, '16'): 'K',
    (25, '12'): 'G',
    (25, '13'): 'H',
    (25, '19'): 'B',
    (25, '23'): 'K',
    (26, '3'): 'B',
    (26, '4'): 'H',
    (26, '5'): 'G',
    (26, '14'): 'K',
    (27, '3'): 'B',
    (27, '4'): 'H',
    (27, '5'): 'G',
    (27, '14'): 'K',
    (28, '8'): 'B',
    (28, '11'): 'K',
    (28, '803'): 'G',
    (28, '804'): 'H',
    (29, '3'): 'H',
    (29, '4'): 'G',
    (29, '9'): 'B',
    (29, '16'): 'K',
    (30, '3'): 'H',
    (30, '4'): 'G',
    (30, '14'): 'K',
    (31, '12'): 'G',
    (31, '13'): 'H',
    (31, '19'): 'B',
    (31, '23'): 'K',
    (32, '3'): 'B',
    (32, '4'): 'H',
    (32, '5'): 'G',
    (32, '14'): 'K',
    (33, '3'): 'H',
    (33, '4'): 'G',
    (33, '10'): 'B',
    (33, '15'): 'K',
    (34, '13'): 'G',
    (34, '21'): 'B',
    (34, '25'): 'K',
    (35, '4'): 'H',
    (35, '5'): 'G',
    (35, '14'): 'K',
    (36, '3'): 'B',
    (36, '4'): 'H',
    (36, '5'): 'G',
    (36, '16'): 'K',
    (37, '3'): 'H',
    (37, '4'): 'G',
    (37, '9'): 'B',
    (37, '14'): 'K',
    (38, '4'): 'H',
    (38, '5'): 'G',
    (38, '10'): 'B',
    (38, '15'): 'K',
    (39, '4'): 'H',
    (39, '5'): 'G',
    (39, '14'): 'B',
    (39, '18'): 'K',
    (40, '4'): 'B',
    (40, '5'): 'H',
    (40, '6'): 'G',
    (40, '15'): 'K',
    (41, '4'): 'G',
    (41, '5'): 'H',
    (41, '13'): 'B',
    (41, '17'): 'K',
    (42, '3'): 'B',
    (42, '4'): 'H',
    (42, '5'): 'G',
    (42, '15'): 'K',
    (43, '13'): 'G',
    (43, '14'): 'H',
    (43, '22'): 'B',
    (43, '26'): 'K',
    (44, '3'): 'G',
    (44, '18'): 'B',
    (44, '22'): 'K',
    (45, '9'): 'B',
    (45, '14'): 'K',
    (46, '3'): 'B',
    (46, '5'): 'G',
    (46, '15'): 'K',
    (47, '3'): 'B',
    (47, '5'): 'G',
    (47, '15'): 'K',
    (48, '9'): 'B',
    (48, '14'): 'K',
    (49, '3'): 'B',
    (49, '4'): 'H',
    (49, '14'): 'K',
    (50, '5'): 'B',
    (50, '9'): 'K',
    (50, '503'): 'G',
    (50, '504'): 'H',
    (51, '5'): 'B',
    (51, '9'): 'K',
    (51, '503'): 'G',
    (51, '504'): 'H',
    (52, '3'): 'H',
    (52, '4'): 'G',
    (52, '9'): 'B',
    (52, '14'): 'K',
    (53, '3'): 'B',
    (53, '4'): 'H',
    (53, '5'): 'G',
    (54, '5'): 'H',
    (54, '6'): 'G',
    (54, '9'): 'B',
    (54, '13'): 'K',
    (55, '4'): 'H',
    (55, '5'): 'G',
    (55, '11'): 'B',
    (55, '16'): 'K',
    (56, '4'): 'B',
    (56, '5'): 'H',
    (56, '16'): 'K',
    (57, '5'): 'H',
    (57, '6'): 'G',
    (57, '15'): 'K',
    (58, '4'): 'H',
    (58, '5'): 'G',
    (58, '15'): 'K',
    (59, '4'): 'B',
    (59, '5'): 'H',
    (59, '6'): 'G',
    (59, '16'): 'K',
    (60, '10'): 'C',
    (60, '705'): 'G',
    (60, '706'): 'H',
    (60, '904'): 'O',
    (61, '22'): 'B',
    (61, '27'): 'K',
    (61, '2205'): 'G',
    (61, '2206'): 'H',
    (62, '3'): 'B',
    (62, '4'): 'H',
    (62, '5'): 'G',
    (62, '15'): 'K',
    (63, '8'): 'B',
    (63, '16'): 'K',
    (64, '4'): 'G',
    (64, '6'): 'B',
    (64, '13'): 'K',
    (65, '3'): 'G',
    (65, '13'): 'K',
    (66, '14'): 'K',
    (67, '16'): 'K',
    (68, '11'): 'B',
    (69, '15'): 'K',
    (70, '5'): 'G',
    (70, '7'): 'B',
    (70, '14'): 'K',
    (71, '6'): 'G',
    (71, '8'): 'B',
    (71, '16'): 'K',
    (72, '5'): 'G',
    (72, '7'): 'B',
    (72, '14'): 'K',
    (73, '5'): 'G',
    (73, '7'): 'B',
    (73, '14'): 'K',
    (74, '5'): 'G',
    (74, '7'): 'B',
    (74, '14'): 'K',
    (75, '5'): 'G',
    (75, '7'): 'B',
    (75, '14'): 'K',
    (76, '5'): 'G',
    (76, '7'): 'B',
    (76, '14'): 'K',
    (77, '5'): 'G',
    (77, '7'): 'B',
    (77, '21'): 'K',
    (78, '6'): 'G',
    (78, '8'): 'B',
    (78, '16'): 'K',
    (79, '6'): 'G',
    (79, '8'): 'B',
    (79, '17'): 'K',
    (80, '5'): 'G',
    (80, '7'): 'B',
    (80, '17'): 'K',
    (81, '10'): 'C',
    (82, '6'): 'G',
    (82, '8'): 'B',
    (82, '15'): 'K',
    (83, '7'): 'G',
    (83, '9'): 'B',
    (84, '6'): 'G',
    (84, '8'): 'B',
    (84, '18'): 'K',
    (85, '5'): 'G',
    (85, '7'): 'B',
    (85, '14'): 'K',
    (86, '5'): 'G',
    (86, '7'): 'B',
    (86, '14'): 'K',
    (87, '14'): 'C',
    (87, '16'): 'K',
    (87, '603'): 'G',
    (88, '27'): 'G',
    (88, '29'): 'B',
    (88, '36'): 'K',
    (89, '29'): 'G',
    (89, '31'): 'B',
    (89, '60'): 'E',
    (90, '6'): 'G',
    (90, '8'): 'B',
    (90, '12'): 'K',
    (91, '4'): 'G',
    (91, '7'): 'B',
    (91, '14'): 'K',
    (92, '29'): 'G',
    (92, '31'): 'B',
    (92, '43'): 'E',
    (93, '5'): 'G',
    (93, '7'): 'B',
    (93, '21'): 'K',
    (94, '29'): 'G',
    (94, '31'): 'B',
    (94, '38'): 'K',
    (95, '29'): 'G',
    (95, '31'): 'B',
    (95, '38'): 'K',
}

# Mapping of Labels to Channel Identifiers for nnU-Net (will be updated dynamically)
LABEL_TO_CHANNEL = {
    'B': '0000',  # Example: Channel 0
    'H': '0001',  # Example: Channel 1
    'G': '0002',  # Example: Channel 2
    'K': '0003',  # Example: Channel 3
}

# -----------------------------
# Define Eligible Patients
# -----------------------------

def get_eligible_patients(dataset_dir):
    """
    Automatically detects eligible patient IDs by listing directories in the dataset directory.

    Parameters:
    - dataset_dir: Path to the dataset's parent directory.

    Returns:
    - List of patient IDs as strings.
    """
    try:
        # List all entries in the dataset directory
        all_entries = os.listdir(dataset_dir)
        # Filter out only directories that match the four-digit patient ID format
        eligible_patients = [entry for entry in all_entries 
                             if os.path.isdir(os.path.join(dataset_dir, entry)) and entry.isdigit() and len(entry) == 4]
        print(f"Found {len(eligible_patients)} eligible patients.")
        return eligible_patients
    except Exception as e:
        print(f"Error detecting eligible patients: {e}")
        return []

# Initialize ELIGIBLE_PATIENTS
ELIGIBLE_PATIENTS = get_eligible_patients(DATASET_DIR)

# -----------------------------
# Helper Functions
# -----------------------------

def convert_dicom_to_nifti(dicom_dir, output_path, reference_image=None, is_mask=False):
    """
    Converts a directory of DICOM files to a NIfTI (.nii.gz) file.
    If a reference_image is provided, resamples the output to match it.

    Parameters:
    - dicom_dir: Path to the directory containing DICOM files.
    - output_path: Path where the NIfTI file will be saved.
    - reference_image: SimpleITK Image to match spatial properties.
    - is_mask: Boolean indicating if the image is a mask (affects interpolation).

    Returns:
    - Converted SimpleITK Image if successful, None otherwise.
    """
    try:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
        if not dicom_names:
            print(f"No DICOM files found in {dicom_dir}.")
            return None
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        print(f"Converted DICOM to NIfTI: {output_path}")

        if reference_image:
            if not images_have_same_spatial_properties(image, reference_image):
                print(f"  [Info] Resampling {output_path} to match reference image spatial properties.")
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(reference_image)
                resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
                image = resampler.Execute(image)

        sitk.WriteImage(image, output_path)
        return image
    except Exception as e:
        print(f"Error converting DICOM in {dicom_dir}: {e}")
        return None

def images_have_same_spatial_properties(image1, image2):
    """
    Checks if two SimpleITK images have the same spatial properties.

    Parameters:
    - image1, image2: SimpleITK Image objects.

    Returns:
    - True if same spatial properties, False otherwise.
    """
    return (image1.GetSize() == image2.GetSize() and
            image1.GetSpacing() == image2.GetSpacing() and
            image1.GetOrigin() == image2.GetOrigin() and
            image1.GetDirection() == image2.GetDirection())

def create_dataset_json(dataset_folder, channel_names, labels, num_training):
    """
    Creates a dataset.json file for nnU-Net.

    Parameters:
    - dataset_folder: Path to the dataset folder.
    - channel_names: Dictionary mapping channel indices to names.
    - labels: Dictionary mapping label names to integers.
    - num_training: Number of training cases.
    """
    dataset_json = {
        "channel_names": {str(k): v for k, v in channel_names.items()},
        "labels": {k: v for k, v in labels.items()},
        "numTraining": num_training,
        "file_ending": FILE_ENDING
    }

    try:
        with open(os.path.join(dataset_folder, 'dataset.json'), 'w') as f:
            json.dump(dataset_json, f, indent=4)
        print(f"Created dataset.json at {os.path.join(dataset_folder, 'dataset.json')}")
    except Exception as e:
        print(f"Error creating dataset.json: {e}")

def merge_masks(mask_paths, output_mask_path):
    """
    Merges multiple mask NIfTI files into a single mask using the maximum value.
    Resamples masks to match the first mask's spatial properties if necessary.

    Parameters:
    - mask_paths: List of paths to mask NIfTI files.
    - output_mask_path: Path to save the merged mask.

    Returns:
    - True if merging is successful, False otherwise.
    """
    try:
        combined_mask = None
        reference_mask = sitk.ReadImage(mask_paths[0])
        reference_size = reference_mask.GetSize()
        reference_spacing = reference_mask.GetSpacing()
        reference_origin = reference_mask.GetOrigin()
        reference_direction = reference_mask.GetDirection()

        for mask_path in mask_paths:
            mask = sitk.ReadImage(mask_path)
            print(f"    [Debug] Mask: {mask_path}, Shape: {mask.GetSize()}")
            if mask.GetSize() != reference_size or mask.GetSpacing() != reference_spacing or \
               mask.GetOrigin() != reference_origin or mask.GetDirection() != reference_direction:
                print(f"      [Info] Resampling {mask_path} to match reference mask spatial properties.")
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(reference_mask)
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # For masks
                mask = resampler.Execute(mask)
            mask_array = sitk.GetArrayFromImage(mask)
            if combined_mask is None:
                combined_mask = mask_array
            else:
                combined_mask = np.maximum(combined_mask, mask_array)
        # Convert back to SimpleITK Image
        final_mask = sitk.GetImageFromArray(combined_mask)
        final_mask.CopyInformation(reference_mask)  # Copy spatial info from reference mask
        sitk.WriteImage(final_mask, output_mask_path)
        print(f"    [Info] Merged and saved mask: {output_mask_path}")
        return True
    except Exception as e:
        print(f"    [Error] Error merging masks: {e}")
        return False

def load_dlss_mapping(csv_path):
    """
    Loads the DLDS mapping from a CSV file.

    Parameters:
    - csv_path: Path to the CSV file.

    Returns:
    - Dictionary mapping PatientID to DLDS.
    """
    try:
        df = pd.read_csv(csv_path)
        # Strip any leading/trailing whitespace from PatientID
        df['PatientID'] = df['PatientID'].astype(str).str.strip()
        # Ensure PatientID is a four-digit string
        df['PatientID'] = df['PatientID'].apply(lambda x: x.zfill(4))
        # Create the mapping
        mapping = pd.Series(df.DLDS.values, index=df.PatientID).to_dict()
        print(f"Loaded DLDS mapping for {len(mapping)} patients.")
        return mapping
    except Exception as e:
        print(f"Error loading DLDS mapping CSV: {e}")
        return {}

def determine_top_modalities(eligible_patients, dataset_dir, dlss_mapping, series_to_label, top_n=3):
    """
    Determines the top N most common modalities across all eligible patients.

    Parameters:
    - eligible_patients: List of eligible patient IDs.
    - dataset_dir: Path to the dataset's parent directory.
    - dlss_mapping: Dictionary mapping PatientID to DLDS.
    - series_to_label: Dictionary mapping (DLDS, Series Number) to Label.
    - top_n: Number of top modalities to select.

    Returns:
    - List of top N labels.
    """
    label_counter = Counter()
    print("\nDetermining the top {} most common modalities...".format(top_n))
    for patient_id in eligible_patients:
        padded_patient_id = patient_id.zfill(4)
        patient_folder = os.path.join(dataset_dir, padded_patient_id)
        if padded_patient_id not in dlss_mapping:
            print(f"  [Warning] DLDS for patient {padded_patient_id} not found. Skipping.")
            continue
        dlss = dlss_mapping.get(padded_patient_id, None)
        if dlss is None:
            print(f"  [Warning] DLDS for patient {padded_patient_id} is None. Skipping.")
            continue
        # Process series folders
        series_folders = [f for f in os.listdir(patient_folder) if os.path.isdir(os.path.join(patient_folder, f))]
        for series_folder in series_folders:
            try:
                series_num = series_folder.split('_')[0]
            except IndexError:
                print(f"    [Warning] Unexpected folder naming in {series_folder}. Skipping this folder.")
                continue
            label = series_to_label.get((dlss, series_num), None)
            if label:
                label_counter[label] += 1
    top_labels = [label for label, count in label_counter.most_common(top_n)]
    print(f"Top {top_n} modalities: {top_labels}")
    return top_labels

# -----------------------------
# Main Processing Function
# -----------------------------

def main():
    # Load DLDS mapping
    dlss_mapping = load_dlss_mapping(DLDS_MAPPING_CSV)
    if not dlss_mapping:
        print("DLDS mapping is empty. Please check the CSV file.")
        return

    # Determine the top 3 most common modalities
    top_modalities = determine_top_modalities(ELIGIBLE_PATIENTS, DATASET_DIR, dlss_mapping, SERIES_TO_LABEL, top_n=3)
    if not top_modalities:
        print("No modalities found. Exiting.")
        return

    # Update LABEL_TO_CHANNEL to include only the top 3 modalities
    # Assign '0000', '0001', '0002' to the top modalities in order
    updated_label_to_channel = {label: f"{i:04d}" for i, label in enumerate(top_modalities)}
    print(f"Updated LABEL_TO_CHANNEL: {updated_label_to_channel}")

    # Define paths for the new dataset
    dataset_folder = os.path.join(NNUNET_RAW_DIR, f"Dataset{DATASET_ID:03d}_{DATASET_NAME}")
    imagesTr_folder = os.path.join(dataset_folder, 'imagesTr')
    labelsTr_folder = os.path.join(dataset_folder, 'labelsTr')

    # Create necessary directories
    os.makedirs(imagesTr_folder, exist_ok=True)
    os.makedirs(labelsTr_folder, exist_ok=True)
    print(f"Created directories: {imagesTr_folder} and {labelsTr_folder}")

    # Define labels mapping for dataset.json
    labels_mapping = {
        "background": 0,
        "liver": 1  # Update based on your segmentation labels
    }

    # Initialize a counter for successful training cases
    successful_cases = 0

    # Iterate through each eligible patient
    for idx, patient_id in enumerate(ELIGIBLE_PATIENTS, start=1):
        print(f"\nProcessing patient {idx}/{len(ELIGIBLE_PATIENTS)}: {patient_id}")

        # Ensure PatientID is a four-digit string
        padded_patient_id = patient_id.zfill(4)

        # Define the patient folder path
        patient_folder = os.path.join(DATASET_DIR, padded_patient_id)

        # Debug: Check if PatientID exists in dlss_mapping
        if padded_patient_id in dlss_mapping:
            print(f"  [Info] DLDS for patient {padded_patient_id}: {dlss_mapping[padded_patient_id]}")
        else:
            print(f"  [Warning] DLDS for patient {padded_patient_id} not found. Skipping.")
            continue

        dlss = dlss_mapping.get(padded_patient_id, None)
        if dlss is None:
            print(f"  [Warning] DLDS for patient {padded_patient_id} is None. Skipping.")
            continue

        # Define a unique case identifier (e.g., Liver_0003)
        case_identifier = f"Liver_{padded_patient_id}"

        # Initialize a dictionary to hold modality file paths
        modalities = {}

        # Initialize a list to hold all mask NIfTI paths
        mask_nifti_paths = []

        # Reference image for resampling
        reference_image = None

        # Process each series folder (e.g., '3_randomnumber', '4_randomnumber', etc.)
        series_folders = [f for f in os.listdir(patient_folder) if os.path.isdir(os.path.join(patient_folder, f))]
        if not series_folders:
            print(f"  [Warning] No series folders found in {patient_folder}. Skipping patient.")
            continue

        for series_folder in series_folders:
            print(f"    Processing series folder: {series_folder}")
            # Extract the first number before the underscore (e.g., '3' from '3_874')
            try:
                series_num = series_folder.split('_')[0]
                print(f"      [Info] Series Number: {series_num}")
            except IndexError:
                print(f"      [Warning] Unexpected folder naming in {series_folder}. Skipping this folder.")
                continue

            # Map (DLDS, Series Number) to Label
            label = SERIES_TO_LABEL.get((dlss, series_num), None)
            if label is None:
                print(f"      [Warning] Series number '{series_num}' in DLDS '{dlss}' is not mapped to any label. Skipping this series.")
                continue

            # Check if the label is among the top modalities
            if label not in top_modalities:
                print(f"      [Info] Label '{label}' is not among the top modalities {top_modalities}. Skipping this series.")
                continue

            # Check if this label has already been processed for this patient
            if label in modalities:
                print(f"      [Warning] Label '{label}' already processed for patient {padded_patient_id}. Skipping this series.")
                continue

            print(f"      [Info] Mapped Label: {label}")

            # Get the channel identifier
            channel_id = updated_label_to_channel.get(label, None)
            if channel_id is None:
                print(f"      [Warning] Label '{label}' does not have a corresponding channel identifier. Skipping this series.")
                continue
            print(f"      [Info] Channel ID: {channel_id}")

            # Define paths to 'images_randomnumber' and 'masks_randomnumber' folders
            images_glob = glob(os.path.join(patient_folder, series_folder, 'images_*'))
            masks_glob = glob(os.path.join(patient_folder, series_folder, 'masks_*'))

            if not images_glob:
                print(f"      [Warning] No 'images_*' folder found in {os.path.join(patient_folder, series_folder)}. Skipping this series.")
                continue
            if not masks_glob:
                print(f"      [Warning] No 'masks_*' folder found in {os.path.join(patient_folder, series_folder)}. Skipping this series.")
                continue

            images_folder_path = images_glob[0]
            masks_folder_path = masks_glob[0]
            print(f"      [Info] Images folder: {images_folder_path}")
            print(f"      [Info] Masks folder: {masks_folder_path}")

            # Define the output image and mask paths in temporary filenames
            temp_image_path = os.path.join(imagesTr_folder, f"{case_identifier}_{channel_id}_temp{FILE_ENDING}")
            temp_mask_path = os.path.join(labelsTr_folder, f"{case_identifier}_{channel_id}_temp{FILE_ENDING}")

            # Convert images DICOM to NIfTI
            print(f"      [Info] Converting images to NIfTI...")
            image = convert_dicom_to_nifti(images_folder_path, temp_image_path, reference_image=reference_image, is_mask=False)
            if image:
                modalities[label] = temp_image_path  # Store temp path
                if reference_image is None:
                    reference_image = image  # Set reference_image to the first converted image
            else:
                print(f"      [Error] Failed to convert images in '{images_folder_path}' for patient {padded_patient_id}. Skipping patient.")
                # Clean up any temporary files that have been created so far
                for temp_path in modalities.values():
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                for temp_mask in mask_nifti_paths:
                    if os.path.exists(temp_mask):
                        os.remove(temp_mask)
                modalities = None
                break  # Exit the loop and skip this patient

            # Convert masks DICOM to NIfTI and add to mask paths
            print(f"      [Info] Converting masks to NIfTI...")
            mask_image = convert_dicom_to_nifti(masks_folder_path, temp_mask_path, reference_image=reference_image, is_mask=True)
            if mask_image:
                mask_nifti_paths.append(temp_mask_path)
            else:
                print(f"      [Error] Failed to convert masks in '{masks_folder_path}' for patient {padded_patient_id}. Skipping patient.")
                # Clean up any temporary files that have been created so far
                for temp_path in modalities.values():
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                mask_nifti_paths = []
                modalities = None
                break  # Exit the loop and skip this patient

        if modalities is None:
            print(f"  [Error] Skipping patient {padded_patient_id} due to image conversion failures.")
            continue

        if not mask_nifti_paths:
            print(f"  [Error] Skipping patient {padded_patient_id} due to mask conversion failures.")
            # Clean up any temporary files
            for temp_path in modalities.values():
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            continue

        # Check if all top_modalities are present
        patient_modalities = set(modalities.keys())
        if not set(top_modalities).issubset(patient_modalities):
            print(f"  [Info] Patient {padded_patient_id} does not have all top modalities {top_modalities}. Skipping.")
            # Clean up temporary files
            for temp_path in modalities.values():
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            for temp_mask in mask_nifti_paths:
                if os.path.exists(temp_mask):
                    os.remove(temp_mask)
            continue

        # Define the final image and mask paths
        for label, temp_image_path in modalities.items():
            final_image_path = os.path.join(imagesTr_folder, f"{case_identifier}_{updated_label_to_channel[label]}{FILE_ENDING}")
            shutil.move(temp_image_path, final_image_path)
            print(f"    [Info] Saved image: {final_image_path}")

        # Merge multiple masks into a single mask if necessary
        if len(mask_nifti_paths) > 1:
            print(f"    [Info] Merging {len(mask_nifti_paths)} mask files...")
            final_mask_path = os.path.join(labelsTr_folder, f"{case_identifier}{FILE_ENDING}")
            success = merge_masks(mask_nifti_paths, final_mask_path)
            if success:
                # Remove individual mask files after merging
                for temp_mask in mask_nifti_paths:
                    try:
                        os.remove(temp_mask)
                    except FileNotFoundError:
                        print(f"    [Warning] Mask file {temp_mask} not found during cleanup.")
                print(f"    [Info] Merged mask saved at {final_mask_path}")
            else:
                print(f"    [Error] Failed to merge masks for patient {padded_patient_id}. Skipping patient.")
                # Clean up saved images
                for label in top_modalities:
                    final_image_path = os.path.join(imagesTr_folder, f"{case_identifier}_{updated_label_to_channel[label]}{FILE_ENDING}")
                    if os.path.exists(final_image_path):
                        os.remove(final_image_path)
                continue
        else:
            # Rename the single mask file to the required format
            single_mask_path = mask_nifti_paths[0]
            final_mask_path = os.path.join(labelsTr_folder, f"{case_identifier}{FILE_ENDING}")
            shutil.move(single_mask_path, final_mask_path)
            print(f"    [Info] Saved mask: {final_mask_path}")

        # Increment the successful cases counter
        successful_cases += 1
        print(f"  [Success] Processed patient {padded_patient_id} successfully.")

    # -----------------------------
    # Finalize dataset.json
    # -----------------------------

    # Define channel names for dataset.json
    # nnU-Net expects channel indices as strings
    # Assign channel names based on the labels
    # For example, if top_modalities = ['H', 'G', 'B'], then:
    # '0': 'H', '1': 'G', '2': 'B'
    channel_names = {str(i): label for i, label in enumerate(top_modalities)}

    # Define labels mapping for dataset.json
    # This should reflect the segmentation labels used
    labels_mapping = {
        "background": 0,
        "liver": 1  # Update based on your segmentation labels
    }

    # Set num_training to successful_cases
    num_training = successful_cases
    print(f"\nTotal successful training cases: {num_training}")

    # Create dataset.json
    create_dataset_json(dataset_folder, channel_names, labels_mapping, num_training)

    print("\nDataset conversion completed successfully.")

# -----------------------------
# Execute the Main Function
# -----------------------------
if __name__ == "__main__":
    main()
