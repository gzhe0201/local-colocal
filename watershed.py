from aicsimageio import AICSImage
import numpy as np
import napari
import os
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_objects, binary_closing, ball
from skimage.segmentation import watershed, clear_border
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

def watershed_czi(czi_path, dapi_channel=0, dapi_thresh = 15, min_voxels=50, seed_distance=0,
                  smoothing_sigma=5.0, peak_threshold=5):
    
    # --- Load CZI and get DAPI channel ---
    img = AICSImage(czi_path)
    dapi = img.get_image_data("ZYX", C=dapi_channel)

    # --- Smooth image for thresholding ---
    dapi = gaussian(dapi, sigma=smoothing_sigma, preserve_range=True)

    # --- Threshold to binary ---
    binary = dapi > dapi_thresh

    filled_slices = np.zeros_like(binary) 
    for z in range(binary.shape[0]): 
        filled_slices[z] = ndi.binary_fill_holes(binary[z])
        binary[z] = filled_slices[z]

    # --- Clean mask ---
    binary = remove_small_objects(binary, min_size=min_voxels)

    # --- Distance transform ---
    distance = ndi.distance_transform_edt(binary)

    # --- Heavily smooth distance to merge lumps ---
    smoothed_distance = gaussian(distance, sigma=smoothing_sigma, preserve_range=True)

    # --- Find seed coordinates ---
    coordinates = peak_local_max(
        smoothed_distance,
        footprint=np.ones((60,60,60)),
        labels=binary,
        min_distance=seed_distance,
        threshold_abs=peak_threshold
    )
    print(f"Found {len(coordinates)} seeds for watershed.")

    # --- Convert coordinates to markers array ---
    mask = np.zeros_like(distance, dtype=bool)
    if len(coordinates) > 0:
        mask[tuple(coordinates.T)] = True
    markers, _ = ndi.label(mask)

    # --- Run 3D watershed ---
    labeled_mask = watershed(-distance, markers, mask=binary)

    # --- Remove edge-nuclei post-watershed ---
    labeled_mask = remove_xy_edge_clusters(labeled_mask)
    print(f"Watershed produced {labeled_mask.max()} labels.")

    # --- Napari visualization ---
    viewer = napari.Viewer()
    viewer.add_image(binary, name='binary', colormap='gray')
    viewer.add_labels(labeled_mask, name='labels')
    napari.run()

    return labeled_mask


def remove_xy_edge_clusters(labeled, adjacency_threshold=20):
    cleaned = labeled.copy()

    # --- Find labels touching the XY borders ---
    edge_labels = np.unique(np.concatenate([
        labeled[:, 0, :],        # y = 0
        labeled[:, -1, :],       # y = max
        labeled[:, :, 0],        # x = 0
        labeled[:, :, -1]        # x = max
    ]))
    edge_labels = edge_labels[edge_labels != 0]

    if len(edge_labels) == 0:
        return cleaned  # no edges found, nothing to remove

    # --- Find directly adjacent labels (one layer) ---
    structure = np.ones((3, 3, 3), dtype=bool)
    edge_mask = np.isin(labeled, edge_labels)
    dilated = ndi.binary_dilation(edge_mask, structure=structure, iterations=1)
    touching_labels = np.unique(labeled[dilated])
    touching_labels = touching_labels[(touching_labels != 0) & (~np.isin(touching_labels, edge_labels))]

    # --- Measure contact area for each touching label ---
    labels_to_remove = set(edge_labels.tolist())
    edge_border_mask = dilated & (~edge_mask)

    # --- Remove them ---
    for lbl in touching_labels:
        lbl_mask = (labeled == lbl)
        contact_voxels = np.sum(lbl_mask & edge_border_mask)
        if contact_voxels >= adjacency_threshold:
            labels_to_remove.add(lbl)

    # --- Remove selected labels ---
    mask_to_remove = np.isin(cleaned, list(labels_to_remove))
    cleaned[mask_to_remove] = 0

    # Optional visualization
    viewer = napari.Viewer()
    viewer.add_labels(labeled, name='Original Labels')
    viewer.add_labels(cleaned, name='Cleaned Labels')
    napari.run()

    return cleaned




# === MAIN EXECUTION ===
if __name__ == "__main__":
    input_dir = input("Enter input directory path: ").strip('"')
    output_dir = input("Enter output directory path: ").strip('"')

    # Get all .czi files in the folder
    czi_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".czi")
    ]

    if not czi_files:
        print("⚠️ No .czi files found in the input directory.")
    else:
        print(f"Found {len(czi_files)} .czi file(s). Starting processing...\n")

    # Loop through each .czi file
    for i, filename in enumerate(czi_files, start=1):
        czi_path = os.path.join(input_dir, filename)
        print(f"[{i}/{len(czi_files)}] Processing: {filename}")

        try:
            mask = watershed_czi(czi_path)

            # Optionally save or visualize mask here if desired
            # For example:
            # np.save(os.path.join(output_dir, f"{filename}_mask.npy"), mask)

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    print("\n✅ Processing complete.")

