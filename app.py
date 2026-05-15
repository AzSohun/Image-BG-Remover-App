from google.colab import files
import cv2
import numpy as np
import re

# ---------- Upload image ----------
uploaded = files.upload()
file_name = list(uploaded.keys())[0]

# ---------- Read image ----------
img = cv2.imread(file_name)

if img is None:
    raise ValueError("Image not loaded properly")

# ---------- Convert to RGB ----------
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---------- GrabCut segmentation ----------
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

h, w = img.shape[:2]
rect = (10, 10, w - 20, h - 20)

cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)

# ---------- Create transparent image ----------
img_rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
img_rgba[:, :, 3] = mask2 * 255

# ---------- Safe filename function ----------
def clean_filename(name):
    name = name.strip()
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    if not name.endswith(".png"):
        name += ".png"
    return name

# ---------- User filename input ----------
image_name = input("Enter output file name (e.g. my_image.png): ")
image_name = clean_filename(image_name)

print("Saving as:", image_name)

# ---------- Save image ----------
cv2.imwrite(image_name, cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGRA))

print("Done! File saved:", image_name)

# ---------- Download ----------
files.download(image_name)