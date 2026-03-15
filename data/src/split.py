import os
import random
import shutil

source_dir = "data/raw/ultrasound_images"
destination_dir = "data/processed/ultrasound_split"

classes = ["normal", "abnormal"]

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

for cls in classes:

    cls_path = os.path.join(source_dir, cls)
    images = os.listdir(cls_path)

    random.shuffle(images)

    total = len(images)

    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    for split_name, split_images in zip(
        ["train", "val", "test"],
        [train_images, val_images, test_images]
    ):

        split_path = os.path.join(destination_dir, split_name, cls)

        os.makedirs(split_path, exist_ok=True)

        for img in split_images:

            src = os.path.join(cls_path, img)
            dst = os.path.join(split_path, img)

            shutil.copy(src, dst)

print("Dataset successfully split into train, val, test!")