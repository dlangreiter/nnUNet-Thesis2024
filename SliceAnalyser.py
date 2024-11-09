# performance_metrics_visualizer.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import nibabel as nib
import os
import sys
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Button, CheckButtons, Slider

# Attempt to import roc_auc_score from scikit-learn
try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    print("Error: scikit-learn is required for computing AUROC. Please install it using 'pip install scikit-learn'")
    sys.exit(1)

def load_nifti_image(nifti_path):
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

def load_nifti_mask(nifti_path, binarize=True):
    data, affine, header = load_nifti_image(nifti_path)
    if binarize:
        mask = (data > 0).astype(np.uint8)
    else:
        mask = data  # Assume continuous scores for prediction
    print(f"Loaded mask from {nifti_path}: unique values {np.unique(mask)}")
    return mask, affine, header

def print_voxel_spacing(header, label):
    voxel_spacing = header.get_zooms()[:3]
    print(f"{label} Voxel Spacing (mm): {voxel_spacing}")

def check_affine_alignment(affine_gt, affine_pred, affine_anat):
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

def dice_coefficient(gt, pred):
    intersection = np.logical_and(gt, pred).sum()
    gt_sum = gt.sum()
    pred_sum = pred.sum()
    if gt_sum + pred_sum == 0:
        return 1.0  # Both masks are empty
    return 2. * intersection / (gt_sum + pred_sum)

def auroc_score_func(gt, pred_scores):
    gt_flat = gt.flatten()
    pred_flat = pred_scores.flatten()
    # Check if both classes are present
    if np.unique(gt_flat).size == 1:
        print("Warning: Only one class present in ground truth. AUROC is not defined.")
        return np.nan
    try:
        auroc = roc_auc_score(gt_flat, pred_flat)
        return auroc
    except ValueError as e:
        print(f"Error computing AUROC: {e}")
        return np.nan

def create_combined_mask(ground_truth, prediction):
    combined_mask = np.zeros_like(ground_truth, dtype=np.uint8)
    combined_mask += ground_truth  # 1 where ground truth mask is present
    # Threshold prediction scores at 0.5 to get binary prediction mask
    pred_binary = (prediction > 0.5).astype(np.uint8)
    combined_mask += pred_binary * 2  # 2 where prediction mask is present
    return combined_mask

def display_masks_slice_combined_with_anatomical(anatomical_image, gt_mask, pred_mask, overlap_mask, slice_idx, cmap_gt, cmap_pred, cmap_overlap, ax):
    if slice_idx < 0 or slice_idx >= anatomical_image.shape[2]:
        print(f"Warning: Slice index {slice_idx} is out of bounds for data with {anatomical_image.shape[2]} slices.")
        ax.set_axis_off()
        ax.set_title(f'Slice {slice_idx} (Invalid Index)')
        return None, None, None

    anatomical_slice = anatomical_image[:, :, slice_idx]
    gt_slice = gt_mask[:, :, slice_idx]
    pred_slice = pred_mask[:, :, slice_idx]
    overlap_slice = overlap_mask[:, :, slice_idx]

    # Print unique values to verify masks
    print(f"Slice {slice_idx}: Ground Truth unique values: {np.unique(gt_slice)}")
    print(f"Slice {slice_idx}: Prediction unique values: {np.unique((pred_slice > 0.5).astype(np.uint8))}")
    print(f"Slice {slice_idx}: Overlap unique values: {np.unique(overlap_slice)}")

    # Normalise anatomical image for better contrast
    anat_norm = anatomical_slice / np.max(anatomical_slice) if np.max(anatomical_slice) != 0 else anatomical_slice

    # Display anatomical image in grayscale
    img = ax.imshow(anat_norm, cmap='gray', interpolation='none')

    # Overlay Ground Truth Mask
    gt_overlay = ax.imshow(gt_slice, cmap=cmap_gt, alpha=0.5, interpolation='none', vmin=0, vmax=1)

    # Overlay Prediction Mask
    pred_overlay = ax.imshow(pred_slice, cmap=cmap_pred, alpha=0.5, interpolation='none', vmin=0, vmax=1)

    # Overlay Overlap Mask
    overlap_overlay = ax.imshow(overlap_slice, cmap=cmap_overlap, alpha=0.5, interpolation='none', vmin=0, vmax=1)

    ax.axis('off')
    ax.set_title(f'Slice {slice_idx}')

    return gt_overlay, pred_overlay, overlap_overlay

def display_masks_grid_with_anatomical(anatomical_image, ground_truth, prediction, cmap_gt, cmap_pred, cmap_overlap, slice_indices, grid_size=(3,3), save_path=None):
    rows, cols = grid_size
    total_slices = rows * cols
    num_slices = anatomical_image.shape[2]

    if len(slice_indices) != total_slices:
        print(f"Error: Number of slice indices provided ({len(slice_indices)}) does not match grid size ({total_slices}).")
        sys.exit(1)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axes = axes.flatten()

    for idx, slice_idx in enumerate(slice_indices):
        ax = axes[idx]
        gt_overlay, pred_overlay, overlap_overlay = display_masks_slice_combined_with_anatomical(
            anatomical_image,
            ground_truth,
            prediction,
            ground_truth & (prediction > 0.5).astype(np.uint8),  # Overlap
            slice_idx,
            cmap_gt,
            cmap_pred,
            cmap_overlap,
            ax
        )
        # Store overlays if needed
        axes[idx].gt_overlay = gt_overlay
        axes[idx].pred_overlay = pred_overlay
        axes[idx].overlap_overlay = overlap_overlay

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

def create_interactive_slideshow(anatomical_image, ground_truth, prediction, cmap_gt, cmap_pred, cmap_overlap, interval=500, save=False, save_path='slideshow.gif'):
    num_slices = anatomical_image.shape[2]
    current_slice = [0]  # Use list for mutable integer in nested functions
    playing = [False]    # Use list for mutable boolean in nested functions

    # Calculate Overlap
    overlap = ground_truth & (prediction > 0.5).astype(np.uint8)  # Overlap where both masks are present

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.35)  # Make space for buttons and controls

    # Initialise the first frame
    gt_overlay, pred_overlay, overlap_overlay = display_masks_slice_combined_with_anatomical(
        anatomical_image,
        ground_truth,
        prediction,
        overlap,
        current_slice[0],
        cmap_gt,
        cmap_pred,
        cmap_overlap,
        ax
    )
    # Create a legend
    red_patch = mpatches.Patch(color='red', label='Ground Truth')
    green_patch = mpatches.Patch(color='green', label='Prediction')
    yellow_patch = mpatches.Patch(color='yellow', label='Overlap')
    black_patch = mpatches.Patch(color='black', label='Background')
    # Position the legend outside the main plot
    fig.legend(handles=[red_patch, green_patch, yellow_patch, black_patch],
               loc='upper right', bbox_to_anchor=(1.15, 1))

    # Define button axes
    axprev = plt.axes([0.1, 0.25, 0.1, 0.04])  # [left, bottom, width, height]
    axpause = plt.axes([0.21, 0.25, 0.1, 0.04])
    axnext = plt.axes([0.32, 0.25, 0.1, 0.04])

    # Create buttons
    bprev = Button(axprev, 'Previous')
    bpause = Button(axpause, 'Play')
    bnext = Button(axnext, 'Next')

    # Define CheckButtons axes
    axcheck = plt.axes([0.05, 0.05, 0.15, 0.15])  # [left, bottom, width, height]
    check = CheckButtons(axcheck, ['Ground Truth', 'Prediction', 'Overlap'], [True, True, True])

    # Define Slider axes
    axslider_gt = plt.axes([0.3, 0.15, 0.4, 0.03])
    axslider_pred = plt.axes([0.3, 0.10, 0.4, 0.03])
    axslider_overlap = plt.axes([0.3, 0.05, 0.4, 0.03])

    slider_gt = Slider(axslider_gt, 'GT Alpha', 0.0, 1.0, valinit=0.5)
    slider_pred = Slider(axslider_pred, 'Pred Alpha', 0.0, 1.0, valinit=0.5)
    slider_overlap = Slider(axslider_overlap, 'Overlap Alpha', 0.0, 1.0, valinit=0.5)

    # Text annotations for metrics
    dice_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    auroc_text = ax.text(0.05, 0.90, '', transform=ax.transAxes, fontsize=12,
                         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    intensity_text = ax.text(0.05, 0.85, '', transform=ax.transAxes, fontsize=12,
                             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))  # New Text for Intensity

    # Update functions
    def update_display():
        anatomical_slice = anatomical_image[:, :, current_slice[0]]
        gt_slice = ground_truth[:, :, current_slice[0]]
        pred_slice = prediction[:, :, current_slice[0]]
        overlap_slice = overlap[:, :, current_slice[0]]

        # Print unique values to verify masks
        print(f"Displaying Slice {current_slice[0]}:")
        print(f" - Ground Truth unique values: {np.unique(gt_slice)}")
        print(f" - Prediction unique values: {np.unique((pred_slice > 0.5).astype(np.uint8))}")
        print(f" - Overlap unique values: {np.unique(overlap_slice)}")

        # Normalise anatomical image for better contrast
        anat_norm = anatomical_slice / np.max(anatomical_slice) if np.max(anatomical_slice) != 0 else anatomical_slice

        # Update anatomical image
        img = ax.images[0]
        img.set_data(anat_norm)

        # Update masks
        gt_overlay.set_data(gt_slice)
        pred_overlay.set_data(pred_slice)
        overlap_overlay.set_data(overlap_slice)

        # Compute Dice Coefficient
        dice = dice_coefficient(gt_slice, (pred_slice > 0.5).astype(np.uint8))
        dice_text.set_text(f'Dice Coefficient: {dice:.2f}')

        # Compute AUROC
        auroc = auroc_score_func(gt_slice, pred_slice)
        if np.isnan(auroc):
            auroc_text.set_text('AUROC: N/A')
        else:
            auroc_text.set_text(f'AUROC: {auroc:.2f}')

        # Compute Average Intensity
        avg_intensity = anatomical_slice.mean()
        intensity_text.set_text(f'Avg Intensity: {avg_intensity:.2f}')

        # Update titles
        ax.set_title(f'Slice {current_slice[0]}')

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

    def on_check(label):
        if label == 'Ground Truth':
            gt_overlay.set_visible(not gt_overlay.get_visible())
        elif label == 'Prediction':
            pred_overlay.set_visible(not pred_overlay.get_visible())
        elif label == 'Overlap':
            overlap_overlay.set_visible(not overlap_overlay.get_visible())
        fig.canvas.draw_idle()

    def update_alpha_gt(val):
        gt_overlay.set_alpha(val)
        fig.canvas.draw_idle()

    def update_alpha_pred(val):
        pred_overlay.set_alpha(val)
        fig.canvas.draw_idle()

    def update_alpha_overlap(val):
        overlap_overlay.set_alpha(val)
        fig.canvas.draw_idle()

    # Connect button events
    bnext.on_clicked(next_slice)
    bprev.on_clicked(prev_slice)
    bpause.on_clicked(play_pause)

    # Connect checkboxes
    check.on_clicked(on_check)

    # Connect sliders
    slider_gt.on_changed(update_alpha_gt)
    slider_pred.on_changed(update_alpha_pred)
    slider_overlap.on_changed(update_alpha_overlap)

    # Timer for auto-play
    def update_frame(frame):
        if playing[0]:
            if current_slice[0] < num_slices - 1:
                current_slice[0] += 1
                update_display()
            else:
                playing[0] = False
                bpause.label.set_text('Play')

    ani = animation.FuncAnimation(fig, update_frame, frames=num_slices, interval=interval, blit=False)

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
    # Paths to the NIfTI files
    anatomical_image_path = '/home/declan/thesis7/10percenttest/input/BRAT_208_0000.nii.gz'  
    ground_truth_path = '/home/declan/thesis7/10percenttest/ground_truth/BRAT_208.nii.gz'     
    prediction_path = '/home/declan/thesis7/resultPostprocess/BRAT_208.nii.gz'                

    slice_indices = [50, 100, 150, 200, 250, 300, 350, 400, 450] 

    # Visualisation options
    visualize_grid = False     
    visualize_slideshow = True 
    grid_size = (3, 3)          
    interval = 500               
    save_slideshow = False      
    save_path = 'slideshow.gif'  

    # ===================================================================

    # Load images
    anatomical_image, affine_anat, header_anat = load_nifti_image(anatomical_image_path)
    ground_truth, affine_gt, header_gt = load_nifti_mask(ground_truth_path, binarize=True)
    prediction, affine_pred, header_pred = load_nifti_mask(prediction_path, binarize=False)  # Assume continuous scores

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

    # Define separate masks for overlays
    overlap = ground_truth & (prediction > 0.5).astype(np.uint8)  # Overlap where both masks are present

    # Define colormaps for each mask with proper transparency
    cmap_gt = ListedColormap(['#00000000', '#FF0000FF'])       # Ground Truth: Transparent and Red
    cmap_pred = ListedColormap(['#00000000', '#00FF00FF'])    # Prediction: Transparent and Green
    cmap_overlap = ListedColormap(['#00000000', '#FFFF00FF']) # Overlap: Transparent and Yellow

    # Visualisation Logic
    if visualize_grid:
        # Ensure exactly 9 slice indices are provided
        if len(slice_indices) != grid_size[0] * grid_size[1]:
            print(f"Error: Number of slice indices provided ({len(slice_indices)}) does not match grid size ({grid_size[0] * grid_size[1]}).")
            sys.exit(1)
        # Display grid
        display_masks_grid_with_anatomical(
            anatomical_image,
            ground_truth,
            prediction,
            cmap_gt,
            cmap_pred,
            cmap_overlap,
            slice_indices,
            grid_size=grid_size,
            save_path=None  
        )

    if visualize_slideshow:
        # Create and display interactive slideshow
        create_interactive_slideshow(
            anatomical_image,
            ground_truth,
            prediction,
            cmap_gt,
            cmap_pred,
            cmap_overlap,
            interval=interval,
            save=save_slideshow,
            save_path=save_path
        )

    if not visualize_grid and not visualize_slideshow:
        print("No visualization mode selected. Exiting.")
        sys.exit(0)

if __name__ == "__main__":
    main()
