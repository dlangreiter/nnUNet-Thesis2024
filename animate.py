import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)  # Create three subplots side by side

# Load the NIfTI files
nifti_file_input = nib.load('/home/declan/thesis3/test1/BRAT_001_0000.nii.gz')
data_input = nifti_file_input.get_fdata()

nifti_file_ground_truth = nib.load('/home/declan/thesis3/nifti_and_segms/TCGA-BC-4073/02-21-2000/rater2_liver.nii.gz')
data_ground_truth = nifti_file_ground_truth.get_fdata()

# Load a third NIfTI file
nifti_file_third = nib.load('/home/declan/thesis3/postprocessing_output/BRAT_001.nii.gz')  # Replace with the actual path to your third NIfTI file
data_third = nifti_file_third.get_fdata()

def animate(i):
    ax1.clear()  # Clear the previous frame in the first subplot
    ax1.imshow(data_input[:, :, i], cmap='gray')  # Display the current frame in the first subplot
    ax1.set_title(f"Input Frame {i+1}")  # Set the title of the first subplot

    ax2.clear()  # Clear the previous frame in the second subplot
    ax2.imshow(data_ground_truth[:, :, i], cmap='gray')  # Display the current frame in the second subplot
    ax2.set_title(f"Ground Truth Frame {i+1}")  # Set the title of the second subplot

    ax3.clear()  # Clear the previous frame in the third subplot
    ax3.imshow(data_third[:, :, i], cmap='gray')  # Display the current frame in the third subplot
    ax3.set_title(f"nnUNet Seg {i+1}")  # Set the title of the third subplot

# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=data_input.shape[2], interval=400)

plt.show()
