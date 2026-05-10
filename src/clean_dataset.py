import os
import shutil
import cv2
from tqdm import tqdm
try:
    import pytesseract
except ImportError:
    pytesseract = None

# Configuration
RAW_DIR = os.path.join("data", "raw", "ultrasound_images")
CLEAN_DIR = os.path.join("data", "clean")
REJECTED_DIR = os.path.join("data", "rejected", "non_bladder")

# Keywords
REJECT_FILENAME_KEYWORDS = [
    "kidney", "liver", "ovary", "prostate", "cyst", "spleen", "kl", "rt ov", "lc ov",
    "gallbladder", "pancreas", "uterus", "rectum", "colon", "bowel", "intestine",
    "aorta", "appendix", "ovarian", "hepatic", "renal", "splenic", "adrenal"
]
REJECT_OCR_KEYWORDS = [
    "KIDNEY", "LIVER", "OVARY", "PROSTATE", "SPLEEN", "GALLBLADDER", "UTERUS",
    "PANCREAS", "HEPATIC", "RENAL", "LT OV", "RT OV", "KL", "KR", "LK", "RK"
]
KEEP_KEYWORDS = ["bladder", "ub", "urinary", "bl", "vb", "vesica"]

def check_tesseract():
    if pytesseract is None:
        return False
    # Check if command is available
    try:
        pytesseract.get_tesseract_version()
        return True
    except pytesseract.TesseractNotFoundError:
        return False
    except Exception:
        return False

def clean_dataset():
    print("=== Starting Dataset Cleaning ===")
    
    # Ensure directories exist
    for split in ["normal", "abnormal"]:
        os.makedirs(os.path.join(CLEAN_DIR, split), exist_ok=True)
        os.makedirs(os.path.join(REJECTED_DIR, split), exist_ok=True)
    
    tesseract_available = check_tesseract()
    if not tesseract_available:
        print("WARNING: Tesseract OCR is not installed or not found in PATH.")
        print("Falling back to filename-only filtering. Skipping OCR step.")
        
    stats = {
        "normal_before": 0, "normal_removed": 0, "normal_kept": 0,
        "abnormal_before": 0, "abnormal_removed": 0, "abnormal_kept": 0
    }
    
    for split in ["normal", "abnormal"]:
        src_folder = os.path.join(RAW_DIR, split)
        if not os.path.exists(src_folder):
            print(f"Directory not found: {src_folder}")
            continue
            
        images = [f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        stats[f"{split}_before"] = len(images)
        
        for filename in tqdm(images, desc=f"Cleaning {split} images"):
            filepath = os.path.join(src_folder, filename)
            filename_lower = filename.lower()
            
            # Step 3 prioritisation (if filename contains keep keywords)
            if any(k.lower() in filename_lower for k in KEEP_KEYWORDS):
                # Keep immediately
                shutil.copy(filepath, os.path.join(CLEAN_DIR, split, filename))
                stats[f"{split}_kept"] += 1
                continue
            
            # Step 1: Filter by filename keywords
            if any(k.lower() in filename_lower for k in REJECT_FILENAME_KEYWORDS):
                shutil.copy(filepath, os.path.join(REJECTED_DIR, split, filename))
                stats[f"{split}_removed"] += 1
                continue
                
            # Step 2: Filter by OCR
            ocr_rejected = False
            if tesseract_available:
                try:
                    img = cv2.imread(filepath)
                    if img is not None:
                        text = pytesseract.image_to_string(img).upper()
                        if any(k in text for k in REJECT_OCR_KEYWORDS):
                            ocr_rejected = True
                except Exception as e:
                    pass # Ignore read errors
                    
            if ocr_rejected:
                shutil.copy(filepath, os.path.join(REJECTED_DIR, split, filename))
                stats[f"{split}_removed"] += 1
                continue
                
            # Step 3 default (kept)
            shutil.copy(filepath, os.path.join(CLEAN_DIR, split, filename))
            stats[f"{split}_kept"] += 1
            
    # Step 4: Final report
    print("\n=== Dataset Cleaning Report ===")
    print(f"Normal images before:  {stats['normal_before']}")
    print(f"Normal images removed: {stats['normal_removed']}  (non-bladder)")
    print(f"Normal images kept:    {stats['normal_kept']}")
    print("")
    print(f"Abnormal images before:  {stats['abnormal_before']}")
    print(f"Abnormal images removed: {stats['abnormal_removed']}  (non-bladder)")
    print(f"Abnormal images kept:    {stats['abnormal_kept']}")
    print("")
    total_clean = stats['normal_kept'] + stats['abnormal_kept']
    print(f"Total clean images: {total_clean}")
    print(f"Removed images saved to: {REJECTED_DIR}")
    print(f"Clean images saved to:   {CLEAN_DIR}")

if __name__ == "__main__":
    clean_dataset()
