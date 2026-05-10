import os
import shutil
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from tqdm import tqdm

CLEAN_DIR = os.path.join("data", "clean")
BALANCED_DIR = os.path.join("data", "balanced")
RESULTS_DIR = os.path.join("results", "graphs")

TARGET_PER_CLASS = 2000

# Augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=12, p=0.6),               # max 12 degrees only
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    A.GaussianBlur(blur_limit=3, p=0.3),
    A.GaussNoise(var_limit=(5, 20), p=0.3),  # simulates ultrasound speckle
    A.Affine(scale=(0.92, 1.08), translate_percent=0.04, p=0.4),
    A.CLAHE(clip_limit=2.0, p=0.4),
    A.ElasticTransform(alpha=1, sigma=4, p=0.2),
])

def balance_dataset():
    print("=== Starting Dataset Balancing ===")
    
    os.makedirs(os.path.join(BALANCED_DIR, "normal"), exist_ok=True)
    os.makedirs(os.path.join(BALANCED_DIR, "abnormal"), exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    stats = {}
    preview_images = []
    
    for split in ["normal", "abnormal"]:
        src_folder = os.path.join(CLEAN_DIR, split)
        dest_folder = os.path.join(BALANCED_DIR, split)
        
        images = [f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        original_count = len(images)
        stats[f"{split}_original"] = original_count
        
        if original_count == 0:
            stats[f"{split}_augmented"] = 0
            stats[f"{split}_total"] = 0
            continue
            
        print(f"Processing {split} (current: {original_count}, target: {TARGET_PER_CLASS})")
        
        if original_count > TARGET_PER_CLASS:
            # Sample down
            selected = random.sample(images, TARGET_PER_CLASS)
            for f in tqdm(selected, desc=f"Copying {split}"):
                shutil.copy(os.path.join(src_folder, f), os.path.join(dest_folder, f))
            stats[f"{split}_augmented"] = 0
            stats[f"{split}_total"] = TARGET_PER_CLASS
            
        else:
            # Copy all originals first
            for f in tqdm(images, desc=f"Copying {split}"):
                shutil.copy(os.path.join(src_folder, f), os.path.join(dest_folder, f))
                
            # Augment remainder
            needed = TARGET_PER_CLASS - original_count
            stats[f"{split}_augmented"] = needed
            stats[f"{split}_total"] = TARGET_PER_CLASS
            
            pbar = tqdm(total=needed, desc=f"Augmenting {split}")
            aug_count = 0
            while aug_count < needed:
                # Randomly pick a source image to augment
                src_img_name = random.choice(images)
                src_path = os.path.join(src_folder, src_img_name)
                
                img = cv2.imread(src_path)
                if img is None:
                    continue
                # OpenCV loads as BGR, convert to RGB for albumentations
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                augmented = transform(image=img_rgb)['image']
                
                # Convert back to BGR for saving
                aug_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                
                # Save
                aug_filename = f"aug_{split}_{aug_count:04d}.jpg"
                aug_path = os.path.join(dest_folder, aug_filename)
                cv2.imwrite(aug_path, aug_bgr)
                
                # Save some for preview
                if len(preview_images) < 16:
                    preview_images.append(augmented)
                    
                aug_count += 1
                pbar.update(1)
            pbar.close()

    # Create preview grid
    if preview_images:
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        fig.suptitle("Augmentation Preview", fontsize=16)
        for i, ax in enumerate(axes.flat):
            if i < len(preview_images):
                ax.imshow(preview_images[i])
                ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "augmentation_preview.png"))
        plt.close()

    print("\n=== Balanced Dataset Report ===")
    print(f"Normal images (original):    {stats.get('normal_original', 0)}")
    print(f"Normal images (augmented):   {stats.get('normal_augmented', 0)}")
    print(f"Normal images (total):       {stats.get('normal_total', 0)}")
    print("")
    print(f"Abnormal images (original):  {stats.get('abnormal_original', 0)}")
    print(f"Abnormal images (augmented): {stats.get('abnormal_augmented', 0)}")
    print(f"Abnormal images (total):     {stats.get('abnormal_total', 0)}")
    print("")
    grand_total = stats.get('normal_total', 0) + stats.get('abnormal_total', 0)
    print(f"Grand total: {grand_total} images")
    if stats.get('abnormal_total', 0) > 0 and stats.get('normal_total', 0) > 0:
        ratio = stats.get('normal_total', 0) / stats.get('abnormal_total', 0)
        print(f"Class balance ratio: {ratio:.2f}:1  (perfectly balanced)")
    print("")
    print(f"Saved to: {BALANCED_DIR}")

if __name__ == "__main__":
    balance_dataset()
