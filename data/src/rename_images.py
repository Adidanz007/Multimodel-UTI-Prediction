import os

dataset_path = "data/raw/ultrasound_images"

classes = ["normal", "abnormal"]

for cls in classes:

    class_path = os.path.join(dataset_path, cls)

    files = os.listdir(class_path)

    files.sort()

    for i, filename in enumerate(files):

        old_path = os.path.join(class_path, filename)

        extension = os.path.splitext(filename)[1]

        new_name = f"{cls}_{i+1:04d}{extension}"

        new_path = os.path.join(class_path, new_name)

        os.rename(old_path, new_path)

print("Renaming completed successfully!")