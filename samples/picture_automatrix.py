import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from skimage import io
from skimage.morphology import remove_small_objects, opening, disk
from skimage.measure import label, regionprops
from skimage.filters import median
from skimage.segmentation import find_boundaries
import pandas as pd
import random

# Set random seeds
SEED = 27
np.random.seed(SEED)
random.seed(SEED)
cv2.setNumThreads(1)

def select_crop_area(image, scale=0.2):
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    resized = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    roi = cv2.selectROI("Select ROI", resized, showCrosshair=True)
    cv2.destroyWindow("Select ROI")
    x, y, w, h = [int(val / scale) for val in roi]
    return x, y, w, h

# Load and crop
full_img = cv2.imread('F:/USYD/20250401_0degree1/1 (2500).jpg')
plain_img = cv2.imread('F:/USYD/20250401_0degree1/1 (1).jpg') # reference image (without gas)
x, y, w, h = select_crop_area(full_img, scale=0.2)

full_crop = full_img[y:y+h, x:x+w].astype(np.float32)
plain_crop = plain_img[y:y+h, x:x+w].astype(np.float32)
diff_crop = full_crop - plain_crop
diff_crop_smooth = gaussian_filter(diff_crop, sigma=0.5)

clip_min, clip_max = np.percentile(diff_crop_smooth, 1), np.percentile(diff_crop_smooth, 99)
diff_crop_clipped = np.clip(diff_crop_smooth, clip_min, clip_max)
diff_crop_norm = (diff_crop_clipped - clip_min) / (clip_max - clip_min) * 255
adjusted_crop = diff_crop_norm.astype(np.uint8)

original_ref = adjusted_crop.copy()
adjusted_crop_gray = cv2.cvtColor(adjusted_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

# Estimate light matrix from current image
grid_rows, grid_cols =8,5
percentile = 95
min_bright_pixel_count = 10
tolerance = 10
h, w = adjusted_crop_gray.shape
cell_h, cell_w = h // grid_rows, w // grid_cols
light_values = np.full((grid_rows, grid_cols), np.nan)
valid_mask = np.zeros((grid_rows, grid_cols), dtype=bool)

for r in range(grid_rows):
    for c in range(grid_cols):
        y0, y1 = r * cell_h, (r + 1) * cell_h
        x0, x1 = c * cell_w, (c + 1) * cell_w
        cell_gray = adjusted_crop_gray[y0:y1, x0:x1].flatten()
        if cell_gray.size == 0:
            continue
        thresh = np.percentile(cell_gray, percentile)
        bright_pixels = cell_gray[cell_gray >= thresh]
        if len(bright_pixels) >= min_bright_pixel_count:
            light_values[r, c] = np.mean(bright_pixels)
            valid_mask[r, c] = True

# Remove outliers
global_avg = np.nanmean(light_values)
for r in range(grid_rows):
    for c in range(grid_cols):
        if valid_mask[r, c] and abs(light_values[r, c] - global_avg) >= tolerance:
            valid_mask[r, c] = False
            light_values[r, c] = np.nan

# Fill missing values using neighbors
def average_neighbors(r, c, values, valid_mask):
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < grid_rows and 0 <= cc < grid_cols and not np.isnan(values[rr, cc]):
                neighbors.append(values[rr, cc])
    return np.mean(neighbors) if neighbors else np.nan

for r in range(grid_rows):
    for c in range(grid_cols):
        if not valid_mask[r, c]:
            avg_val = average_neighbors(r, c, light_values, valid_mask)
            light_values[r, c] = avg_val

# Interpolate light values to full image
yy, xx = np.mgrid[0:grid_rows, 0:grid_cols]
points = np.array([(r, c) for r in range(grid_rows) for c in range(grid_cols) if not np.isnan(light_values[r, c])])
values = np.array([light_values[r, c] for r, c in points])
filled = griddata(points, values, (yy, xx), method='cubic')
filled = np.nan_to_num(filled, nan=np.nanmean(filled))

zoom_factors = (h / grid_rows, w / grid_cols)
light_full = zoom(filled, zoom_factors, order=3)
light_full = light_full[:h, :w]
light_smooth = gaussian_filter(light_full, sigma=cell_h / 2)

# Normalize image using light correction
normalized = adjusted_crop_gray / (light_smooth + 1e-5)
norm_min, norm_max = normalized.min(), normalized.max()
normalized_scaled = 255 * (normalized - norm_min) / (norm_max - norm_min)
normalized_scaled = normalized_scaled.astype(np.uint8)

# Display the normalized image after applying the light matrix
fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5), dpi=600)

# Before correction
ax2[0].imshow(adjusted_crop_gray, cmap='gray')
ax2[0].set_title('Before Light Correction')
ax2[0].axis('off')

# After correction
ax2[1].imshow(normalized_scaled, cmap='gray')
ax2[1].set_title('After Light Correction (Normalized)')
ax2[1].axis('off')

plt.tight_layout()
plt.show()


Image.fromarray((normalized * 255).clip(0, 255).astype(np.uint8)).save("normalized_image.png")
Image.fromarray(normalized_scaled).save("normalized_image_scaled.png")

# Segment normalized image
blurred = cv2.GaussianBlur(normalized_scaled, (3, 3), 0)
hist_vals_blur, bin_edges_blur = np.histogram(blurred, bins=35, range=(0, 255))
bin_centers_blur = bin_edges_blur[:-1]
peak_blur_val = bin_centers_blur[np.argmax(hist_vals_blur)]
manual_thresh = min(peak_blur_val-7, 248)

binary_mask = blurred < manual_thresh
binary_cleaned = opening(binary_mask, disk(4))
binary_cleaned = remove_small_objects(label(binary_mask), min_size=500) > 0
median_filtered = median(binary_cleaned, disk(12))

fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=600)
ax[0].imshow(cv2.cvtColor(original_ref, cv2.COLOR_BGR2RGB))
ax[0].set_title('Original (Adjusted Crop)')
ax[0].axis('off')
ax[1].imshow(median_filtered, cmap='gray')
ax[1].set_title('Segmentation Mask (Cleaned)')
ax[1].axis('off')
plt.tight_layout()
plt.show()
