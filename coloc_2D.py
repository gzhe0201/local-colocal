def debug_show_threshold(dapi, dapi_smooth, binary, z_slice=None):
    """
    Show DAPI raw, smoothed, and thresholded binary slice.
    """
    if z_slice is None:
        z_slice = dapi.shape[0] // 2  # middle slice

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(dapi[z_slice], cmap='gray')
    axes[0].set_title(f"Raw DAPI (Z={z_slice})")

    axes[1].imshow(dapi_smooth[z_slice], cmap='gray')
    axes[1].set_title("Smoothed DAPI")

    axes[2].imshow(binary[z_slice], cmap='gray')
    axes[2].set_title("Binary Mask")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()



def colocal_measure (input_directory):
    
    # === STORAGE FOR RESULTS ===
    results = []

    # === LOOP THROUGH IMAGES ===
    for filename in os.listdir(input_directory):
        if not filename.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
            continue

        filepath = os.path.join(input_directory, filename)
        img = cv2.imread(filepath)

        if img is None:
            print(f"Skipping unreadable file: {filename}")
            continue

        # Split channels (OpenCV loads in BGR order)
        b, g, r = cv2.split(img)

        # --- Step 1: Make DAPI mask ---
        _, dapi_mask = cv2.threshold(b, dapi_thresh, 255, cv2.THRESH_BINARY)
        dapi_mask = cv2.morphologyEx(dapi_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        dapi_mask = cv2.morphologyEx(dapi_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # Remove tiny DAPI specks
        label_dapi = label(dapi_mask)
        for region in regionprops(label_dapi):
            if region.area < min_dapi_area:
                dapi_mask[label_dapi == region.label] = 0

        # --- Step 2: Threshold red and mask with DAPI ---
        _, red_mask = cv2.threshold(r, red_thresh, 255, cv2.THRESH_BINARY)
        red_within_dapi = cv2.bitwise_and(red_mask, dapi_mask)

        # --- Step 3: Threshold green and mask with DAPI ---
        _, green_mask = cv2.threshold(g, green_thresh, 255, cv2.THRESH_BINARY)
        green_within_dapi = cv2.bitwise_and(green_mask, dapi_mask)

        # --- Step 4: Overlap region ---
        overlap = cv2.bitwise_and(red_within_dapi, green_within_dapi)

        #plt.imshow(overlap, cmap='gray')
        #plt.title("Red-Green Overlap within DAPI")
        #plt.show()


        # --- Step 5: Analyze overlap particles ---
        label_overlap = label(overlap)
        areas = [region.area for region in regionprops(label_overlap) if region.area >= min_rg_area]

        count = len(areas)
        total_area = sum(areas)
        avg_size = np.mean(areas) if count > 0 else 0

        # Record data
        results.append({
            "Image": filename,
            "Count": count,
            "Total Area": total_area,
            "Average Size": avg_size
        })

        print(f"Processed {filename}: {count} overlaps, total area {total_area}")

    return pd.Dataframe(results) # return DataFrame for use in other scripts
