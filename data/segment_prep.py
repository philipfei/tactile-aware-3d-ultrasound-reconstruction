import os
import sys
import numpy as np
import cv2

# ====================== Configuration ======================
# Must match the crop/resample settings in data/prep.py
X_START, Y_START = 217, 33
WIDTH, HEIGHT = 240, 360
RESAMPLE_FACTOR = 4
ROTATE_IMAGE_90 = False

DIR_DATA = os.path.join(os.path.dirname(__file__), 'segment_data')
DIR_IMAGES = os.path.join(DIR_DATA, 'images')
DIR_MASKS = os.path.join(DIR_DATA, 'masks')
DIR_OUTPUT = os.path.join(DIR_DATA, 'processed_res4')
DIR_OUT_IMAGES = os.path.join(DIR_OUTPUT, 'images')
DIR_OUT_MASKS = os.path.join(DIR_OUTPUT, 'masks')

LUT_PATH = os.path.join(os.path.dirname(__file__), 'he_global_lut_res4.npy')
# ============================================================


def main():
    # Load global HE LUT
    he_lut = np.load(LUT_PATH)

    # Create output directories
    os.makedirs(DIR_OUT_IMAGES, exist_ok=True)
    os.makedirs(DIR_OUT_MASKS, exist_ok=True)

    # Discover PNG files (image and mask filenames match 1:1)
    image_files = sorted([
        f for f in os.listdir(DIR_IMAGES)
        if f.endswith('.png')
    ])

    print(f"Found {len(image_files)} image-mask pairs.")
    print(f"Input size: ({HEIGHT}, {WIDTH}) after crop")
    print(f"Output size: ({HEIGHT // RESAMPLE_FACTOR}, {WIDTH // RESAMPLE_FACTOR}) after resize")
    print(f"HE LUT: {LUT_PATH}")
    print(f"Output: {DIR_OUTPUT}/")

    count = 0
    for fname in image_files:
        base = os.path.splitext(fname)[0]
        img_path = os.path.join(DIR_IMAGES, fname)
        msk_path = os.path.join(DIR_MASKS, fname)

        if not os.path.exists(msk_path):
            print(f"  WARNING: mask not found for {fname}, skipping")
            continue

        # Read image and mask as grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # (480, 640) uint8
        msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)  # (480, 640) uint8 {0,255}

        # Crop
        img = img[Y_START:Y_START + HEIGHT, X_START:X_START + WIDTH]  # (360, 240)
        msk = msk[Y_START:Y_START + HEIGHT, X_START:X_START + WIDTH]

        # Resize
        new_w = WIDTH // RESAMPLE_FACTOR   # 60
        new_h = HEIGHT // RESAMPLE_FACTOR  # 90
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Optional rotation (disabled, same as data/prep.py)
        if ROTATE_IMAGE_90:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            msk = cv2.rotate(msk, cv2.ROTATE_90_CLOCKWISE)

        # Apply global HE LUT to image only
        img = cv2.LUT(img, he_lut)

        # Normalise
        img = img.astype(np.float32) / 255.0               # [0, 1]
        msk = (msk > 0).astype(np.float32)                 # {0.0, 1.0}

        # Save
        out_img_path = os.path.join(DIR_OUT_IMAGES, base + '.npy')
        out_msk_path = os.path.join(DIR_OUT_MASKS, base + '.npy')
        np.save(out_img_path, img)
        np.save(out_msk_path, msk)

        count += 1

    print(f"Done. Processed {count} pairs -> {DIR_OUTPUT}/")


if __name__ == '__main__':
    main()
