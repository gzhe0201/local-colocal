import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import napari

from numpy.lib.stride_tricks import sliding_window_view
from datetime import datetime
from aicsimageio import AICSImage
from skimage.measure import label, regionprops_table, regionprops
from skimage.feature import blob_log
from scipy.ndimage import binary_fill_holes
from scipy.spatial import cKDTree
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_objects, ball, binary_closing, binary_opening, binary_erosion
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.feature import peak_local_max


# === CONFIGURABLE PARAMETERS ===
dapi_thresh = 50         # Threshold for blue/DAPI channel
red_thresh = 75         # Threshold for red channel
green_thresh = 75       # Threshold for green channel
min_dapi_area = 50       # Minimum DAPI region size
min_rg_area = 5         # Minimum red/green particle size



def measure_colocalization(img, labeled_dapi,
                                       green_channel, red_channel,
                                       green_thresh, red_thresh,
                                       min_area):
    """
    Measure red and green puncta within each labeled DAPI nucleus.
    Includes true voxel-overlap colocalized puncta and intensity values
    (from the original, unthresholded images) for overlapping voxels.
    """

    # --- Load image channels ---
    green = img.get_image_data("ZYX", C=green_channel)
    red = img.get_image_data("ZYX", C=red_channel)

    nucleus_summaries = []
    puncta_records = []
    coloc_records = []

    total_nuclei = labeled_dapi.max()
    print(f"Measuring colocalization and puncta for {total_nuclei} nuclei...")

    # Prepare global labeled images to accumulate per-nucleus labels (so Napari shows everything)
    labeled_green_global = np.zeros_like(labeled_dapi, dtype=np.int32)
    labeled_red_global = np.zeros_like(labeled_dapi, dtype=np.int32)    
    large_mask_global = np.zeros_like(labeled_dapi, dtype=np.uint16)
    green_offset = 0
    red_offset = 0

    # Also accumulate a global colocalization mask (voxel-level overlap)
    coloc_mask_global = np.zeros_like(labeled_dapi, dtype=bool)

    for nucleus_id in range(1, total_nuclei + 1):
        nucleus_mask = labeled_dapi == nucleus_id
        if np.sum(nucleus_mask) == 0:
            continue

        # --- Apply thresholds ---
        green_vals = green[nucleus_mask]
        red_vals = red[nucleus_mask]

        if np.count_nonzero(green_vals) > 0:
            t_green = np.mean(green_vals) + 5 * np.std(green_vals)
        else:
            t_green = green_thresh  # fallback to manual default

        if np.count_nonzero(red_vals) > 0:
            t_red = np.mean(red_vals) + 5 * np.std(red_vals)
        else:
            t_red = red_thresh

        # --- Apply thresholds inside this nucleus only ---
        if t_green > 40:
            # Only apply threshold if it’s reasonably high
            green_mask = (green > t_green) & nucleus_mask
        else:
            # Otherwise, set an empty mask (no thresholding)
            green_mask = np.zeros_like(nucleus_mask, dtype=bool)
        if t_red > 40 and t_red < 150:
            # Only apply threshold if it’s reasonably high
            red_mask = (red > t_red) & nucleus_mask
        elif t_red > 150:
            # Otherwise, set an empty mask (no thresholding)
            red_mask = (red > 150) & nucleus_mask
        else: 
            red_mask = np.zeros_like(nucleus_mask, dtype=bool)

        green_mask = binary_erosion(green_mask, ball(0))
        red_mask = binary_erosion(red_mask, ball(0))
        
        green_mask = remove_small_objects(green_mask, min_size=min_area)
        red_mask = remove_small_objects(red_mask, min_size=min_area)
        
        
        labeled_green = label(green_mask, connectivity=1)
        labeled_red = label(red_mask, connectivity=1)

        large_green_regions = [r for r in regionprops(labeled_green) if r.area > 300]

        for r in large_green_regions:
            large_mask_global[labeled_green == r.label] = r.label

        for region in large_green_regions:
            region_mask = labeled_green == region.label
            
            # Extract intensity values from the *original* green channel
            region_intensity = green[region_mask]

            # Skip empty or invalid regions
            if region_intensity.size == 0:
                continue

            # Compute the 50th percentile intensity threshold
            cutoff = np.percentile(region_intensity, 0)

            # Keep only voxels brighter than that cutoff
            filtered_mask = region_mask & (green > cutoff)
            filtered_mask = remove_small_objects(filtered_mask, min_size=5)

            # Relabel the remaining disconnected pieces (if any)
            relabeled = label(filtered_mask, connectivity=1)

            # Offset new labels so they don’t overwrite existing ones
            offset = labeled_green.max()
            relabeled[relabeled > 0] += offset

            # Remove the old region and insert the new one
            labeled_green[region_mask] = 0
            labeled_green[relabeled > 0] = relabeled[relabeled > 0]

        large_red_regions = [r for r in regionprops(labeled_red) if r.area > 300]

        for r in large_red_regions:
            large_mask_global[labeled_red == r.label] = r.label

        for region in large_red_regions:
            region_mask = labeled_red == region.label
            
            # Extract intensity values from the *original* red channel
            region_intensity = red[region_mask]

            # Skip empty or invalid regions
            if region_intensity.size == 0:
                continue

            # Compute the 20th percentile intensity threshold
            cutoff = np.percentile(region_intensity, 20)

            # Keep only voxels brighter than that cutoff
            filtered_mask = region_mask & (red > cutoff)
            filtered_mask = remove_small_objects(filtered_mask, min_size=5)

            # Relabel the remaining disconnected pieces (if any)
            relabeled = label(filtered_mask, connectivity=1)

            # Offset new labels so they don’t overwrite existing ones
            offset = labeled_red.max()
            relabeled[relabeled > 0] += offset

            # Remove the old region and insert the new one
            labeled_red[region_mask] = 0
            labeled_red[relabeled > 0] = relabeled[relabeled > 0]

        n_green_puncta = labeled_green.max()
        n_red_puncta = labeled_red.max()

        coloc_mask = green_mask & red_mask
        coloc_voxels = np.sum(coloc_mask)
        
        # --- RED LACUNARITY ---
        try:
            lac_red, vox_red = lacunarity_3d(red_mask, normalize_fill=True)
            lac_red_norm = lac_red / (vox_red ** (1/3)) if vox_red > 0 else np.nan
        except ValueError as e:
            print(f"⚠️ Lacunarity failed for nucleus (red) {nucleus_id}: {e}")
            lac_red = np.nan
            vox_red = 0
            lac_red_norm = np.nan

        # --- GREEN LACUNARITY ---
        try:
            lac_green, vox_green = lacunarity_3d(green_mask, normalize_fill=True)
            lac_green_norm = lac_green / (vox_green ** (1/3)) if vox_green > 0 else np.nan
        except ValueError as e:
            print(f"⚠️ Lacunarity failed for nucleus (green) {nucleus_id}: {e}")
            lac_green = np.nan
            vox_green = 0
            lac_green_norm = np.nan

        # --- COLOCALIZED LACUNARITY ---
        try:
            lac_coloc, vox_coloc = lacunarity_3d(coloc_mask, normalize_fill=True)
            lac_coloc_norm = lac_coloc / (vox_coloc ** (1/3)) if vox_coloc > 0 else np.nan
        except ValueError as e:
            print(f"⚠️ Lacunarity failed for nucleus (colocalized) {nucleus_id}: {e}")
            lac_coloc = np.nan
            vox_coloc = 0
            lac_coloc_norm = np.nan

        if coloc_voxels > 0:
            mean_green_coloc = np.mean(green[coloc_mask])
            mean_red_coloc = np.mean(red[coloc_mask])
        else:
            mean_green_coloc = 0
            mean_red_coloc = 0
        n_voxels = np.sum(nucleus_mask)

            # --- accumulate labeled puncta into global images with unique offsets ---
        if labeled_green.max() > 0:
            # create a re-labeled block where 0 stays 0, nonzero labels are offset
            relabeled_g = labeled_green.copy()
            relabeled_g[relabeled_g > 0] += green_offset
            # write to global image only inside nucleus (should already be restricted)
            labeled_green_global[relabeled_g > 0] = relabeled_g[relabeled_g > 0]
            green_offset += int(labeled_green.max())

        if labeled_red.max() > 0:
            relabeled_r = labeled_red.copy()
            relabeled_r[relabeled_r > 0] += red_offset
            labeled_red_global[relabeled_r > 0] = relabeled_r[relabeled_r > 0]
            red_offset += int(labeled_red.max())

        # accumulate voxel-wise colocalization into a global mask
        coloc_mask_global |= coloc_mask 

        # --- Per-puncta measurements ---
        df_g, df_r = pd.DataFrame(), pd.DataFrame()

        if n_green_puncta > 0:
            props_g = regionprops_table(
                labeled_green, intensity_image=green,
                properties=("label", "area", "centroid", "mean_intensity")
            )
            df_g = pd.DataFrame(props_g)
            df_g["Nucleus_ID"] = nucleus_id
            df_g["Channel"] = "Green"
            puncta_records.append(df_g)

        if n_red_puncta > 0:
            props_r = regionprops_table(
                labeled_red, intensity_image=red,
                properties=("label", "area", "centroid", "mean_intensity")
            )
            df_r = pd.DataFrame(props_r)
            df_r["Nucleus_ID"] = nucleus_id
            df_r["Channel"] = "Red"
            puncta_records.append(df_r)

        # --- TRUE VOXEL OVERLAP + INTENSITY EXTRACTION ---
        n_coloc_puncta = 0
        if n_green_puncta > 0 and n_red_puncta > 0:
            for g_label in range(1, n_green_puncta + 1):
                g_mask = labeled_green == g_label
                overlap_red_labels = np.unique(labeled_red[g_mask])
                overlap_red_labels = overlap_red_labels[overlap_red_labels != 0]

                for r_label in overlap_red_labels:
                    overlap_mask = g_mask & (labeled_red == r_label)
                    if not np.any(overlap_mask):
                        continue
                    # Compute centroid of the overlapping voxels
                    coords = np.argwhere(overlap_mask)
                    centroid_z, centroid_y, centroid_x = np.mean(coords, axis=0)

                    green_vals = green[overlap_mask]
                    red_vals = red[overlap_mask]

                    # Compute stats
                    mean_g = np.mean(green_vals)
                    mean_r = np.mean(red_vals)
                    median_g = np.median(green_vals)
                    median_r = np.median(red_vals)
                    max_g = np.max(green_vals)
                    max_r = np.max(red_vals)
                    area = overlap_mask.sum()
                    corr = np.corrcoef(green_vals, red_vals)[0, 1] if len(green_vals) > 1 else np.nan

                    coloc_records.append({
                        "Nucleus_ID": nucleus_id,
                        "Green_Label": g_label,
                        "Red_Label": r_label,
                        "Overlap_Voxels": int(area),
                        "Mean_Green_Intensity": float(mean_g),
                        "Mean_Red_Intensity": float(mean_r),
                        "Median_Green_Intensity": float(median_g),
                        "Median_Red_Intensity": float(median_r),
                        "Max_Green_Intensity": float(max_g),
                        "Max_Red_Intensity": float(max_r),
                        "Correlation_G_R": float(corr),
                        "centroid-0": centroid_z,
                        "centroid-1": centroid_y,
                        "centroid-2": centroid_x,
                        "Channel": "Coloc"
                    })
                    n_coloc_puncta += 1

        # --- Per-nucleus summary ---
        total_green_voxels = np.sum(green_mask)
        total_red_voxels = np.sum(red_mask)

        avg_green_puncta_size = total_green_voxels / n_green_puncta if n_green_puncta > 0 else 0
        avg_red_puncta_size = total_red_voxels / n_red_puncta if n_red_puncta > 0 else 0
        avg_coloc_puncta_size = coloc_voxels / n_coloc_puncta if n_coloc_puncta > 0 else 0

        green_puncta_intensity = green[green_mask]
        red_puncta_intensity = red[red_mask]

        mean_green_puncta = np.mean(green_puncta_intensity) if green_puncta_intensity.size > 0 else 0
        mean_red_puncta = np.mean(red_puncta_intensity) if red_puncta_intensity.size > 0 else 0

        if n_voxels == 0:
            continue

        nucleus_summaries.append({
            "Nucleus_ID": nucleus_id,
            "Total_Voxels": n_voxels,
            "Num_Green_Puncta": n_green_puncta,
            "Num_Red_Puncta": n_red_puncta,
            "Num_Coloc_Puncta": n_coloc_puncta,
            "Mean_Green_Puncta_Intensity": mean_green_puncta,
            "Mean_Red_Puncta_Intensity": mean_red_puncta,
            "Mean_Green_Coloc_Intensity": mean_green_coloc,
            "Mean_Red_Coloc_Intensity": mean_red_coloc,
            "Avg_Green_Puncta_Size": avg_green_puncta_size,
            "Avg_Red_Puncta_Size": avg_red_puncta_size,
            "Avg_Coloc_Puncta_Size": avg_coloc_puncta_size,
            "Lacunarity_Red": lac_red,
            "Lacunarity_Green": lac_green,
            "Lacunarity_Coloc": lac_coloc,
            "Lacunarity_Red_norm": lac_red_norm,
            "Lacunarity_Green_norm": lac_green_norm,
            "Lacunarity_Coloc_norm": lac_coloc_norm,
        })


    # Assume you already have a labeled 3D mask called labeled_green
    props = regionprops_table(labeled_dapi, properties=('label', 'centroid'))

    # Create a dataframe for convenience
    df = pd.DataFrame(props)

    # Each centroid becomes a point
    points = df[['centroid-0', 'centroid-1', 'centroid-2']].values

    # The text to display (label IDs)
    labels_text = [str(l) for l in df['label']]

    """viewer = napari.Viewer()
    # raw channels
    viewer.add_image(img.get_image_data("ZYX", C=0), name="DAPI", colormap="blue", blending="additive")
    viewer.add_image(img.get_image_data("ZYX", C=1), name="Green", colormap="green", blending="additive")
    viewer.add_image(img.get_image_data("ZYX", C=2), name="Red", colormap="red", blending="additive")

    # label layers (global)
    viewer.add_labels(labeled_dapi, name="Nuclei Labels", opacity=0.35)
    viewer.add_labels(labeled_green_global, name="Green Puncta (global)", opacity=0.6)
    viewer.add_labels(labeled_red_global, name="Red Puncta (global)", opacity=0.6)
    #viewer.add_labels(large_mask_global, name="Large Red Regions (pre-watershed)", opacity=0.5)
    points_layer = viewer.add_points(points, size=3)
    points_layer.face_color = 'transparent'
    points_layer.edge_color = 'white'
    points_layer.text = {
        'string': labels_text,
        'size': 8,
        'color': 'white',
        'anchor': 'center',
    }


    # colocalized voxels as a label layer (0/1)
    coloc_layer = viewer.add_labels(coloc_mask_global.astype(np.int32), name="Colocalized Voxels", opacity=0.7)
    # try to set color for label 1 to yellow (works across many napari versions, fallback if not)
    try:
        coloc_layer.color = {1: 'yellow'}
    except Exception:
        pass

    napari.run()
"""

    # --- Combine results ---
    df_summary = pd.DataFrame(nucleus_summaries)
    df_puncta = pd.concat(puncta_records, ignore_index=True) if puncta_records else pd.DataFrame()
    df_coloc_puncta = pd.DataFrame(coloc_records)

    # --- Compute spatial spreadness (nearest-neighbor metric) ---
    if not df_puncta.empty or not df_coloc_puncta.empty:
        combined = pd.concat([df_puncta, df_coloc_puncta], ignore_index=True)
        df_spread = compute_spreadness(combined, k_values=(3,4,5))

        # Build a list of dicts where each dict is a nucleus row with Spreadness_{Channel}_k{K} keys
        spread_rows = []
        for _, row in df_spread.iterrows():
            nid = int(row["Nucleus_ID"])
            ch = str(row["Channel"])
            # create dict for this row
            d = {"Nucleus_ID": nid}
            # find all spreadness columns present in df_spread for this row (e.g., Spreadness_k3, Spreadness_k4, ...)
            for col in df_spread.columns:
                if col.startswith("Spreadness_k"):
                    # extract k value suffix
                    k = col.split("Spreadness_k")[-1]
                    d[f"Spreadness_{ch}_k{k}"] = row[col]
            spread_rows.append(d)

        # Convert to dataframe and aggregate in case multiple rows per nucleus/channel exist
        if spread_rows:
            df_spread_pivot = pd.DataFrame(spread_rows)
            df_spread_pivot = df_spread_pivot.groupby("Nucleus_ID").first().reset_index()
        else:
            df_spread_pivot = pd.DataFrame(columns=["Nucleus_ID"])

        # Drop Total_Voxels if accidentally present (prevents the _y duplicate)
        df_spread_pivot = df_spread_pivot.drop(columns=["Total_Voxels"], errors="ignore")

        # Merge into summary
        df_summary = df_summary.merge(df_spread_pivot, on="Nucleus_ID", how="left")
    else:
        print("⚠️ No puncta found; skipping spreadness computation.")


    print(f"✅ Completed {len(df_summary)} nuclei.")
    print(f"🧩 Total puncta: {len(df_puncta)} | Colocalized pairs: {len(df_coloc_puncta)}")

    return df_summary, df_puncta, df_coloc_puncta

def normalize_lac(L, vox):
    return L / (vox ** (1/3)) if vox > 0 else np.nan

def lacunarity_3d(mask, box_sizes=None, normalize_fill=True):
    """
    Compute lacunarity Λ(r) for a 3D binary mask using the gliding-box method.

    Parameters
    ----------
    mask : ndarray (3D)
        Binary array where 1 = filled voxel (signal), 0 = empty.
    box_sizes : list or array of ints, optional
        Box edge lengths (in voxels) to compute lacunarity for.
        Default = [2, 4, 8, 16].
    normalize_fill : bool, default=True
        If True, normalizes for total filling fraction (to allow comparison between samples).

    Returns
    -------
    results : dict
        {
            "r": np.array of box sizes,
            "Lambda": np.array of lacunarity values,
            "fill_fraction": float (original filling fraction),
            "Lambda_mean": float (mean Λ across valid scales)
        }
    """

    # --- Sanity checks ---
    if mask.ndim != 3:
        raise ValueError("Input mask must be a 3D array (Z, Y, X).")

    mask = mask.astype(np.float32)
    fill_fraction = np.mean(mask)

    # Normalize filling fraction if desired
    if normalize_fill:
        # target fill fraction (e.g., 0.5 for equal occupancy across samples)
        target_fill = 0.5
        if fill_fraction > 0:
            # Adjust threshold via random subsampling or rescaling
            p = target_fill / fill_fraction
            p = min(p, 1.0)
            rng = np.random.default_rng(seed=42)
            mask = (mask * (rng.random(mask.shape) < p)).astype(np.float32)
        else:
            raise ValueError("Mask is empty; cannot normalize filling fraction.")

    # Default box sizes
    if box_sizes is None:
        # auto-generate powers of 2 up to 1/4 of smallest dimension
        min_dim = min(mask.shape)
        box_sizes = [2**i for i in range(1, int(np.log2(min_dim//4)) + 1)]
    box_sizes = np.array(box_sizes, dtype=int)

    Lambda_vals = []

    for r in box_sizes:
        if any(dim < r for dim in mask.shape):
            continue  # skip if box is larger than dimension

        # Generate sliding windows
        patches = sliding_window_view(mask, (r, r, r))
        # Sum voxels in each box to get "mass"
        masses = patches.sum(axis=(-3, -2, -1)).ravel()

        mu = np.mean(masses)
        sigma2 = np.var(masses)
        if mu > 0:
            Lambda = sigma2 / (mu ** 2) + 1
            Lambda_vals.append(Lambda)
        else:
            Lambda_vals.append(np.nan)

    Lambda_vals = np.array(Lambda_vals)
    Lambda_mean = np.nanmean(Lambda_vals)
    voxel_count = np.count_nonzero(mask)
    return Lambda_mean, voxel_count

def compute_spreadness(df_puncta, k_values=(3,4,5)):
    spreadness_records = []

    for nucleus_id, group in df_puncta.groupby("Nucleus_ID"):
        for channel in group["Channel"].unique():
            subset = group[group["Channel"] == channel]
            coords = subset[["centroid-0", "centroid-1", "centroid-2"]].values
            if len(coords) <= min(k_values):
                # not enough points for even smallest k
                record = {"Nucleus_ID": nucleus_id, "Channel": channel}
                for k in k_values:
                    record[f"Spreadness_k{k}"] = np.nan
                spreadness_records.append(record)
                continue

            tree = cKDTree(coords)
            dists, _ = tree.query(coords, k=max(k_values) + 1)
            # dists[:,0] = 0 (self), so use columns 1..k

            record = {"Nucleus_ID": nucleus_id, "Channel": channel}
            for k in k_values:
                record[f"Spreadness_k{k}"] = np.mean(dists[:, 1:k + 1])
            spreadness_records.append(record)


    return pd.DataFrame(spreadness_records)


def measure_dapi(czi_path, dapi_channel=0):
    """
    Measure 3D nuclear volumes from a DAPI channel in a .czi image.
    """

    # --- Load image ---
    img = AICSImage(czi_path)
    dapi = img.get_image_data("ZYX", C=dapi_channel)  # shape: (Z, Y, X)
    voxel_sizes = img.physical_pixel_sizes  # (Z, Y, X) in microns

    print(f"Loaded {czi_path}: shape={dapi.shape}, voxel size={voxel_sizes}")

    labeled_dapi = watershed_dapi(img)

    # --- Compute volumes ---
    voxel_vol = np.prod([vs for vs in voxel_sizes if vs is not None])  # µm³ per voxel
    results = []

    for nucleus_id in range(1, labeled_dapi.max() + 1):
        n_voxels = np.sum(labeled_dapi == nucleus_id)
        volume_um3 = n_voxels * voxel_vol
        results.append({
            "Nucleus_ID": nucleus_id,
            "Voxel Count": n_voxels,
            "Volume_um3": volume_um3
        })

    df = pd.DataFrame(results)

    return labeled_dapi


def watershed_dapi(img, dapi_channel=0, dapi_thresh = 15, min_voxels=50, seed_distance=0,
                  smoothing_sigma=5.0, peak_threshold=5):
    
    # --- Get DAPI channel ---
    dapi = img.get_image_data("ZYX", C=dapi_channel)

    # --- Smooth image for thresholding ---
    dapi = gaussian(dapi, sigma=smoothing_sigma, preserve_range=True)

    # --- Threshold to binary ---
    binary = dapi > dapi_thresh

    # --- Clean mask ---
    filled_slices = np.zeros_like(binary) 
    for z in range(binary.shape[0]): 
        filled_slices[z] = ndi.binary_fill_holes(binary[z])
        binary[z] = filled_slices[z]

    binary = remove_small_objects(binary, min_size=min_voxels)

    # --- Distance transform ---
    distance = ndi.distance_transform_edt(binary)

    # --- Heavily smooth distance to merge lumps ---
    smoothed_distance = gaussian(distance, sigma=smoothing_sigma, preserve_range=True)

    # --- Find seed coordinates ---
    coordinates = peak_local_max(
        smoothed_distance,
        footprint=np.ones((50,50,50)),
        labels=binary,
        min_distance=seed_distance,
        threshold_abs=peak_threshold
    )

    # --- Convert coordinates to markers array ---
    mask = np.zeros_like(distance, dtype=bool)
    if len(coordinates) > 0:
        mask[tuple(coordinates.T)] = True
    markers, _ = ndi.label(mask)

    # --- Run 3D watershed ---
    labeled_mask = watershed(-distance, markers, mask=binary)

    # --- Remove edge-nuclei clusters post-watershed ---
    labeled_mask = remove_xy_edge_clusters(labeled_mask)

    # --- Napari visualization ---
    """viewer = napari.Viewer()
    viewer.add_image(binary, name='binary', colormap='gray')
    viewer.add_labels(labeled_mask, name='labels')
    napari.run()"""
    
    print(f"Watershed produced {labeled_mask.max()} labels.")

    return labeled_mask


def remove_xy_edge_clusters(labeled):
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

    # --- Combine edge and adjacent labels ---
    to_remove = np.unique(np.concatenate([edge_labels, touching_labels]))

    # --- Remove them ---
    mask_to_remove = np.isin(cleaned, to_remove)
    cleaned[mask_to_remove] = 0

    return cleaned


def parse_metadata_from_filename(filename, known_celltypes, damage_types, repair_types):
    """
    Extract metadata fields from a .czi filename:
    - Cell type
    - Radiation dose
    - Carbon/Photon/Baseline classification
    - Carbon LET (if available)
    - Time
    """
    name = filename.lower().replace("-", "_").replace(" ", "_")

    # --- 1. Cell type ---
    cell_type = next((ct for ct in known_celltypes if ct.lower() in name), "Unknown")
    damage_type = next((dt for dt in damage_types if dt.lower() in name), "Unknown")
    repair_type = next((rt for rt in repair_types if rt.lower() in name), "Unknown")

    # --- 2. Radiation dose ---
    dose_match = re.search(r"(\d+(\.\d+)?)\s*gy", name)
    radiation_dose = f"{dose_match.group(1)}Gy" if dose_match else "N/A"

    # --- 3. Carbon / Photon / Baseline classification ---
    if "baseline" in name or "0gy" in name:
        carbon_or_photon = "baseline"
    elif "let" in name or re.search(r"c\d+", name):
        carbon_or_photon = "carbon"
    elif "photon" in name:
        carbon_or_photon = "photon"
    else:
        carbon_or_photon = "N/A"

    # --- 4. Carbon LET value ---
    pattern = r"LET(\d+)|(\d+)LET|C(\d+)"
    match = re.search(pattern, name, re.IGNORECASE)

    if match:
        carbon_let = next(int(g) for g in match.groups() if g)
    else:
        carbon_let = "N/A"

    # --- 5. Time ---
    time_match = re.search(r"(\d+)\s*(h|hr|hrs|hour|hours|min|mins|minutes)", name)
    time = f"{time_match.group(1)}{time_match.group(2)}" if time_match else "N/A"

    # --- 6. Replicate ID ---
    # Detect trailing underscore + digit(s), but ignore those in LET/C fields
    rep_match = re.search(r"_([0-9]+)$", name)
    replicate_id = int(rep_match.group(1)) if rep_match else "N/A"



    return {
        "Cell_Type": cell_type.upper(),
        "Damage_Type": damage_type.upper(),
        "Repair_Type": repair_type.upper(),
        "Radiation_Dose": radiation_dose,
        "Carbon_or_Photon": carbon_or_photon,
        "Carbon_LET": carbon_let,
        "Time": time,
        "Replicate_ID": replicate_id
    }


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

    all_image_summaries = []
    
    # Loop through each .czi file
    for i, filename in enumerate(czi_files, start=1):
        czi_path = os.path.join(input_dir, filename)
        prefix = os.path.splitext(filename)[0]

        print(f"[{i}/{len(czi_files)}] Processing: {filename}")

        try:
            labeled_dapi = measure_dapi(czi_path)

            img = AICSImage(czi_path)
            df_summary, df_puncta, df_coloc_puncta = measure_colocalization(
                img, labeled_dapi,
                green_channel=1,
                red_channel=2,
                green_thresh=green_thresh,
                red_thresh=red_thresh,
                min_area=min_rg_area,
                #coloc_dist_thresh=2.0  # distance threshold in voxels
            )

            # Save results
            image_output_dir = os.path.join(output_dir, prefix)
            os.makedirs(image_output_dir, exist_ok=True)

            # === Save results inside this subfolder ===
            df_summary.to_csv(os.path.join(image_output_dir, f"{prefix}_summary.csv"), index=False)
            df_puncta.to_csv(os.path.join(image_output_dir, f"{prefix}_puncta.csv"), index=False)
            df_coloc_puncta.to_csv(os.path.join(image_output_dir, f"{prefix}_coloc_puncta.csv"), index=False)

            # --- Compute per-image averages across nuclei ---
            if "Nucleus_ID" in df_summary.columns:
                df_numeric = df_summary.drop(columns=["Nucleus_ID"])
            else:
                df_numeric = df_summary.copy()
            df_summary_mean = df_numeric.mean(numeric_only=True)
            df_summary_mean["Num_Nuclei"] = len(df_summary)
            df_summary_mean["Image_Name"] = prefix

            # === Extract metadata first ===
            known_celltypes=("HN5", "H1299")
            damage_antibodies=("gH2AX", "h2ax")
            repair_antibodies=("53BP1", "Rad51")
            metadata = parse_metadata_from_filename(prefix, known_celltypes, damage_antibodies, repair_antibodies)

            # --- Add metadata columns ---
            for k, v in metadata.items():
                df_summary_mean[k] = v

            # --- Reorder columns so metadata immediately follows Image_Name ---
            meta_order = [
                "Cell_Type",
                "Damage_Type",
                "Repair_Type",
                "Radiation_Dose",
                "Carbon_or_Photon",
                "Carbon_LET",
                "Time",
                "Replicate_ID"
            ]

            # Create a final ordered list of columns
            ordered_cols = (
                ["Image_Name"]
                + meta_order
                + ["Num_Nuclei"]
                + [col for col in df_summary_mean.index if col not in ["Image_Name", "Num_Nuclei"] + meta_order]
            )

            df_summary_mean = df_summary_mean[ordered_cols]

            # Add to combined list
            all_image_summaries.append(df_summary_mean)

            print(f"✅ Saved all results for {filename} in {image_output_dir}\n")

            # Optionally save or visualize mask here if desired
            # For example:
            # np.save(os.path.join(output_dir, f"{filename}_mask.npy"), mask)

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")


    # Save results
    timestamp = datetime.now().strftime("%d-%b-%Y %H-%M-%S")
    output_path = os.path.join(output_dir, f"{timestamp} counts.csv")

     # --- Combine per-image summaries into one overall CSV ---
    if all_image_summaries:
        df_all_images = pd.DataFrame(all_image_summaries)

        overall_path = os.path.join(output_dir, "Overall_Image_Summary.csv")
        
        df_all_images.to_csv(overall_path, index=False)
        print(f"📊 Combined image summary saved to: {overall_path}")
    
    print(f"\n✅ Results saved to: {output_path}")