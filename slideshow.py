import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import nibabel as nib
import os
import sys
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.widgets import Button

def load_nifti_image(nifti_path):
    """
    Loads a NIfTI image file.

    Parameters:
    - nifti_path: str, path to the NIfTI file

    Returns:
    - data: 3D numpy array
    - affine: 2D numpy array, affine transformation matrix
    - header: NIfTI header object
    """
    if not os.path.exists(nifti_path):
        print(f"Error: File not found - {nifti_path}")
        sys.exit(1)
    try:
        nifti_img = nib.load(nifti_path)
        data = nifti_img.get_fdata()
        affine = nifti_img.affine
        header = nifti_img.header
        return data, affine, header
    except Exception as e:
        print(f"Error loading NIfTI file {nifti_path}: {e}")
        sys.exit(1)

def load_nifti_mask(nifti_path):
    """
    Loads a NIfTI mask file and binarizes it.

    Parameters:
    - nifti_path: str, path to the NIfTI file

    Returns:
    - mask: 3D numpy array of type uint8
    - affine: 2D numpy array, affine transformation matrix
    - header: NIfTI header object
    """
    data, affine, header = load_nifti_image(nifti_path)
    # Binarize the mask: assuming mask values > 0 are considered as mask
    mask = (data > 0).astype(np.uint8)
    return mask, affine, header

def print_voxel_spacing(header, label):
    """
    Prints the voxel spacing from the NIfTI header.

    Parameters:
    - header: NIfTI header object
    - label: str, label to identify the image (e.g., 'Anatomical Image')
    """
    voxel_spacing = header.get_zooms()[:3]
    print(f"{label} Voxel Spacing (mm): {voxel_spacing}")

def check_affine_alignment(affine_gt, affine_pred, affine_anat):
    """
    Checks if the affine matrices of anatomical, ground truth, and prediction images match.

    Parameters:
    - affine_gt: 2D numpy array, affine matrix of ground truth
    - affine_pred: 2D numpy array, affine matrix of prediction
    - affine_anat: 2D numpy array, affine matrix of anatomical image

    Returns:
    - bool, True if all affines match within a tolerance, False otherwise
    """
    aligned_gt_anat = np.allclose(affine_gt, affine_anat, atol=1e-5)
    aligned_pred_anat = np.allclose(affine_pred, affine_anat, atol=1e-5)
    
    if aligned_gt_anat and aligned_pred_anat:
        print("All affine matrices match. Images are aligned.")
        return True
    else:
        if not aligned_gt_anat:
            print("Warning: Ground Truth affine does NOT match Anatomical Image affine.")
            print("Ground Truth Affine:\n", affine_gt)
            print("Anatomical Image Affine:\n", affine_anat)
        if not aligned_pred_anat:
            print("Warning: Prediction affine does NOT match Anatomical Image affine.")
            print("Prediction Affine:\n", affine_pred)
            print("Anatomical Image Affine:\n", affine_anat)
        print("Images may not be properly aligned. Consider resampling.")
        return False

def create_combined_mask(ground_truth, prediction):
    """
    Creates a combined mask with distinct labels:
    0 - Background
    1 - Ground Truth only
    2 - Prediction only
    3 - Both Ground Truth and Prediction

    Parameters:
    - ground_truth: 3D numpy array (uint8)
    - prediction: 3D numpy array (uint8)

    Returns:
    - combined_mask: 3D numpy array with combined labels
    """
    combined_mask = np.zeros_like(ground_truth, dtype=np.uint8)
    combined_mask += ground_truth  # 1 where ground truth mask is present
    combined_mask += prediction * 2  # 2 where prediction mask is present
    # Now:
    # 0 - Background
    # 1 - Ground Truth only
    # 2 - Prediction only
    # 3 - Both
    return combined_mask

def display_masks_slice_combined_with_anatomical(anatomical_image, colored_mask, slice_idx, cmap, norm, ax):
    """
    Displays a specific slice of the anatomical image with combined mask overlays.

    Parameters:
    - anatomical_image: 3D numpy array, anatomical image data
    - colored_mask: 3D numpy array, combined mask with labels
    - slice_idx: int, index of the slice to display
    - cmap: matplotlib Colormap object
    - norm: matplotlib Normalize object
    - ax: matplotlib Axes object
    """
    if slice_idx < 0 or slice_idx >= anatomical_image.shape[2]:
        print(f"Warning: Slice index {slice_idx} is out of bounds for data with {anatomical_image.shape[2]} slices.")
        ax.set_axis_off()
        ax.set_title(f'Slice {slice_idx} (Invalid Index)')
        return

    anatomical_slice = anatomical_image[:, :, slice_idx]
    mask_slice = colored_mask[:, :, slice_idx]

    # Normalize anatomical image for better contrast
    anat_norm = anatomical_slice / np.max(anatomical_slice) if np.max(anatomical_slice) != 0 else anatomical_slice

    # Display anatomical image in grayscale
    ax.imshow(anat_norm, cmap='gray', interpolation='none')

    # Overlay combined masks with transparency
    ax.imshow(mask_slice, cmap=cmap, norm=norm, alpha=0.5, interpolation='none')
    ax.axis('off')
    ax.set_title(f'Slice {slice_idx}')

def display_masks_grid_with_anatomical(anatomical_image, colored_mask, slice_indices, cmap=None, norm=None, grid_size=(3,3), save_path=None):
    """
    Displays a grid of slices with combined masks overlaid on the anatomical image and a unified legend.

    Parameters:
    - anatomical_image: 3D numpy array, anatomical image data
    - colored_mask: 3D numpy array, combined mask with labels
    - slice_indices: list or array of 9 slice indices to display
    - cmap: matplotlib Colormap object
    - norm: matplotlib Normalize object
    - grid_size: tuple, number of rows and columns (default: (3,3))
    - save_path: str or None, file path to save the image. If None, the image is not saved.
    """
    rows, cols = grid_size
    total_slices = rows * cols
    num_slices = colored_mask.shape[2]

    if len(slice_indices) != total_slices:
        print(f"Error: Number of slice indices provided ({len(slice_indices)}) does not match grid size ({total_slices}).")
        sys.exit(1)

    # Validate slice indices
    for idx in slice_indices:
        if idx < 0 or idx >= num_slices:
            print(f"Error: Slice index {idx} is out of bounds for data with {num_slices} slices.")
            sys.exit(1)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axes = axes.flatten()

    for idx, slice_idx in enumerate(slice_indices):
        ax = axes[idx]
        display_masks_slice_combined_with_anatomical(anatomical_image, colored_mask, slice_idx, cmap, norm, ax)

    # Create a unified legend
    red_patch = mpatches.Patch(color='red', label='Ground Truth')
    green_patch = mpatches.Patch(color='green', label='Prediction')
    yellow_patch = mpatches.Patch(color='yellow', label='Overlap')
    black_patch = mpatches.Patch(color='black', label='Background')
    # Position the legend outside the grid
    fig.legend(handles=[red_patch, green_patch, yellow_patch, black_patch],
               loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Adjust to make room for the legend

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved grid visualization to {save_path}")
    else:
        plt.show()

def create_interactive_slideshow(anatomical_image, colored_mask, cmap, norm, interval=500, save=False, save_path='slideshow.gif'):
    """
    Creates and displays an interactive slideshow of all slices with anatomical image and mask overlays.
    Allows pausing, and navigating to next or previous slices.

    Parameters:
    - anatomical_image: 3D numpy array, anatomical image data
    - colored_mask: 3D numpy array, combined mask with labels
    - cmap: matplotlib Colormap object
    - norm: matplotlib Normalize object
    - interval: int, delay between frames in milliseconds for auto-play
    - save: bool, whether to save the slideshow as a GIF
    - save_path: str, file path to save the GIF (if save=True)
    """
    num_slices = anatomical_image.shape[2]
    current_slice = [0]  # Use list for mutable integer in nested functions
    playing = [False]    # Use list for mutable boolean in nested functions

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.2)  # Make space for buttons

    anatomical_slice = anatomical_image[:, :, current_slice[0]]
    mask_slice = colored_mask[:, :, current_slice[0]]
    anat_norm = anatomical_slice / np.max(anatomical_slice) if np.max(anatomical_slice) != 0 else anatomical_slice
    img = ax.imshow(anat_norm, cmap='gray', interpolation='none')
    mask = ax.imshow(mask_slice, cmap=cmap, norm=norm, alpha=0.5, interpolation='none')
    ax.axis('off')
    title = ax.set_title(f'Slice {current_slice[0]}')

    # Create a legend
    red_patch = mpatches.Patch(color='red', label='Ground Truth')
    green_patch = mpatches.Patch(color='green', label='Prediction')
    yellow_patch = mpatches.Patch(color='yellow', label='Overlap')
    black_patch = mpatches.Patch(color='black', label='Background')
    # Position the legend outside the main plot
    fig.legend(handles=[red_patch, green_patch, yellow_patch, black_patch],
               loc='upper right', bbox_to_anchor=(1.15, 1))

    # Define button axes
    axprev = plt.axes([0.3, 0.05, 0.1, 0.075])  # [left, bottom, width, height]
    axpause = plt.axes([0.45, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.6, 0.05, 0.1, 0.075])

    # Create buttons
    bprev = Button(axprev, 'Previous')
    bpause = Button(axpause, 'Play')
    bnext = Button(axnext, 'Next')

    def update_display():
        anatomical_slice = anatomical_image[:, :, current_slice[0]]
        mask_slice = colored_mask[:, :, current_slice[0]]
        anat_norm = anatomical_slice / np.max(anatomical_slice) if np.max(anatomical_slice) != 0 else anatomical_slice

        img.set_data(anat_norm)
        mask.set_data(mask_slice)
        title.set_text(f'Slice {current_slice[0]}')
        fig.canvas.draw_idle()

    def next_slice(event):
        if current_slice[0] < num_slices - 1:
            current_slice[0] += 1
            update_display()

    def prev_slice(event):
        if current_slice[0] > 0:
            current_slice[0] -= 1
            update_display()

    def play_pause(event):
        if playing[0]:
            playing[0] = False
            bpause.label.set_text('Play')
        else:
            playing[0] = True
            bpause.label.set_text('Pause')

    bnext.on_clicked(next_slice)
    bprev.on_clicked(prev_slice)
    bpause.on_clicked(play_pause)

    # Timer for auto-play
    def update_frame(frame):
        if playing[0]:
            if current_slice[0] < num_slices - 1:
                current_slice[0] += 1
            else:
                playing[0] = False
                bpause.label.set_text('Play')
            update_display()

    ani = animation.FuncAnimation(fig, update_frame, interval=interval, blit=False)

    if save:
        # Save as GIF (requires pillow)
        try:
            ani.save(save_path, writer='pillow')
            print(f"Slideshow saved as {save_path}")
        except Exception as e:
            print(f"Error saving slideshow: {e}")
    else:
        plt.show()

def main():
    """
    Main function to load images, verify alignment, and visualize them.
    """
    # ==================== User-Defined Variables ====================

    # Paths to the NIfTI files
    anatomical_image_path = '/home/declan/thesis7/10percenttest/input/BRAT_261_0000.nii.gz'  # Replace with your anatomical NIfTI file path
    ground_truth_path = '/home/declan/thesis7/10percenttest/ground_truth/BRAT_261.nii.gz'          # Replace with your ground truth NIfTI file path
    prediction_path = '/home/declan/thesis7/resultPostprocess/BRAT_261.nii.gz'              # Replace with your prediction NIfTI file path

    # Define the list of 9 specific slice indices to display in the grid
    slice_indices = [50, 100, 150, 200, 250, 300, 350, 400, 450]  # Replace with your desired slice indices

    # Visualization options
    visualize_grid = False      # Set to True to visualize a grid of slices
    visualize_slideshow = True  # Set to True to visualize a slideshow of all slices
    grid_size = (3, 3)           # Number of rows and columns in the grid
    interval = 500               # Delay between frames in the slideshow (in milliseconds)
    save_slideshow = False       # Set to True to save the slideshow as a GIF
    save_path = 'slideshow.gif'  # Path to save the slideshow GIF (if save_slideshow is True)

    # ===================================================================

    # Load images
    anatomical_image, affine_anat, header_anat = load_nifti_image(anatomical_image_path)
    ground_truth, affine_gt, header_gt = load_nifti_mask(ground_truth_path)
    prediction, affine_pred, header_pred = load_nifti_mask(prediction_path)

    # Print voxel spacing for all images
    print_voxel_spacing(header_anat, label='Anatomical Image')
    print_voxel_spacing(header_gt, label='Ground Truth')
    print_voxel_spacing(header_pred, label='Prediction')

    # Check voxel spacing consistency
    voxel_spacing_gt = header_gt.get_zooms()[:3]
    voxel_spacing_pred = header_pred.get_zooms()[:3]
    voxel_spacing_anat = header_anat.get_zooms()[:3]

    if (voxel_spacing_gt != voxel_spacing_anat) or (voxel_spacing_pred != voxel_spacing_anat):
        print("Warning: Voxel spacings do not match among images.")
        print(f"Anatomical Image Voxel Spacing: {voxel_spacing_anat}")
        print(f"Ground Truth Voxel Spacing: {voxel_spacing_gt}")
        print(f"Prediction Voxel Spacing: {voxel_spacing_pred}")
    else:
        print("All voxel spacings match.")

    # Verify affine alignment
    aligned = check_affine_alignment(affine_gt, affine_pred, affine_anat)

    if not aligned:
        print("Images are not properly aligned. Consider resampling to align all images spatially.")
        # Optionally, implement resampling here or exit
        # sys.exit(1)
        # For now, proceed but caution is advised
    else:
        print("All images are aligned. Proceeding with visualization.")

    # Check if dimensions match
    if anatomical_image.shape != ground_truth.shape or anatomical_image.shape != prediction.shape:
        print(f"Error: Shape mismatch among images.")
        print(f"Anatomical Image Shape: {anatomical_image.shape}")
        print(f"Ground Truth Shape: {ground_truth.shape}")
        print(f"Prediction Shape: {prediction.shape}")
        sys.exit(1)

    # Create combined mask
    combined_mask = create_combined_mask(ground_truth, prediction)

    # Define colormap and normalization
    cmap = ListedColormap(['black', 'red', 'green', 'yellow'])  # 0: black, 1: red, 2: green, 3: yellow
    bounds = [0, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)

    # Visualization Logic
    if visualize_grid:
        # Ensure exactly 9 slice indices are provided
        if len(slice_indices) != grid_size[0] * grid_size[1]:
            print(f"Error: Number of slice indices provided ({len(slice_indices)}) does not match grid size ({grid_size[0] * grid_size[1]}).")
            sys.exit(1)
        # Display grid
        display_masks_grid_with_anatomical(
            anatomical_image,
            combined_mask,
            slice_indices,
            cmap=cmap,
            norm=norm,
            grid_size=grid_size,
            save_path=None  # Set to a filename to save the grid visualization
        )

    if visualize_slideshow:
        # Create and display interactive slideshow
        create_interactive_slideshow(
            anatomical_image,
            combined_mask,
            cmap=cmap,
            norm=norm,
            interval=interval,
            save=save_slideshow,
            save_path=save_path
        )

    if not visualize_grid and not visualize_slideshow:
        print("No visualization mode selected. Exiting.")
        sys.exit(0)

if __name__ == "__main__":
    main()
