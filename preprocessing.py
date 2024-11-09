import os
import pydicom
import nibabel as nib
import numpy as np

def dicom_to_nifti(dicom_dir, output_file):
    # Load DICOM files
    dicom_files = []
    for f in sorted(os.listdir(dicom_dir)):
        if f.endswith('.dicom'):
            try:
                dcm = pydicom.dcmread(os.path.join(dicom_dir, f))
                dicom_files.append(dcm)
            except Exception as e:
                print(f"Failed to read DICOM file {f}: {e}")
    if not dicom_files:
        print(f"No valid DICOM files found in {dicom_dir}")
        return  # Exit if no DICOM files to process

    # Stack and convert to NIfTI
    data_array = np.stack([file.pixel_array for file in dicom_files])
    nifti_img = nib.Nifti1Image(data_array, affine=np.eye(4))
    nib.save(nifti_img, output_file)
    print(f"Saved NIfTI file at {output_file}")

def process_folders(root_dir, images_output_dir, masks_output_dir):
    count = 1
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'images' in dirnames:
            images_dir = os.path.join(dirpath, 'images')
            output_name = f"BRATS_{count:03d}.nii.gz"
            dicom_to_nifti(images_dir, os.path.join(images_output_dir, output_name))
        if 'masks' in dirnames:
            masks_dir = os.path.join(dirpath, 'masks')
            output_name = f"BRATS_{count:03d}.nii.gz"
            dicom_to_nifti(masks_dir, os.path.join(masks_output_dir, output_name))
            count += 1  # Increment the BRATS ID



path_to_main_folder = r'/home/declan/thesis/DukeLiverMRI/Segmentation'
path_to_save_images = r'/home/declan/thesis/data/input'
path_to_save_masks = r'/home/declan/thesis/data/ground_truth'
