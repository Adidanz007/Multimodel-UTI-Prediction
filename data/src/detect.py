import os
import hashlib

dataset_path = "data/raw/ultrasound_images"

hashes = {}
duplicates = []

def file_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        filepath = os.path.join(root, file)

        try:
            filehash = file_hash(filepath)

            if filehash in hashes:
                duplicates.append((filepath, hashes[filehash]))
            else:
                hashes[filehash] = filepath

        except Exception as e:
            print(f"Error reading {filepath}: {e}")

print("\nDuplicate Images Found:\n")

for dup in duplicates:
    print(dup)

print("\nTotal duplicates:", len(duplicates))