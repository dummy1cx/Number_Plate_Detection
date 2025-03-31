import os
import shutil
import random
from sklearn.model_selection import train_test_split
import yaml

# ------------------------------------------------------------------------
# The code will create a yaml file which acts like a configuration file
# For ultralytics training yolo expects a yaml configuration file format
# -------------------------------------------------------------------------

BASE_DIR = "/Users/abhishekdas/Documents/INM_705_Practice/Datasets"
IMAGES_DIR = os.path.join(BASE_DIR, "images")
LABELS_DIR = os.path.join(BASE_DIR, "labels")

SPLITS = ["train", "val", "test"]
SPLIT_RATIOS = [0.8, 0.1, 0.1]  # 80% train, 10% val, 10% test for model training
CLASS_NAMES = ["number_plate"]  # For this problem the nu class = 1

def make_dirs():
    for split in SPLITS:
        os.makedirs(os.path.join(BASE_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(BASE_DIR, "labels", split), exist_ok=True)
    print("📁 Output folders created.")

def split_and_move_data():
    image_extensions = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    all_images = [f for f in os.listdir(IMAGES_DIR) if f.endswith(image_extensions)]

    print(f"✅ Looking inside: {IMAGES_DIR}")
    print(f"📸 Total image files found: {len(all_images)}")
    print(f"🗂️ Sample files: {all_images[:5]}")

    if len(all_images) == 0:
        print("❌ No images found. Check your IMAGES_DIR path or image file types.")
        return

    train_files, test_files = train_test_split(all_images, test_size=SPLIT_RATIOS[2], random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=SPLIT_RATIOS[1] / (SPLIT_RATIOS[0] + SPLIT_RATIOS[1]), random_state=42)

    split_map = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    for split, files in split_map.items():
        for f in files:
            base_name = os.path.splitext(f)[0]
            img_src = os.path.join(IMAGES_DIR, f)
            label_src = os.path.join(LABELS_DIR, base_name + ".txt")

            img_dst = os.path.join(BASE_DIR, "images", split, f)
            label_dst = os.path.join(BASE_DIR, "labels", split, base_name + ".txt")

            try:
                shutil.copy(img_src, img_dst)
                print(f"✅ Copied image: {img_src} → {img_dst}")
            except Exception as e:
                print(f"❌ Failed to copy image {img_src}: {e}")

            if os.path.exists(label_src):
                try:
                    shutil.copy(label_src, label_dst)
                    print(f"✅ Copied label: {label_src} → {label_dst}")
                except Exception as e:
                    print(f"❌ Failed to copy label {label_src}: {e}")
            else:
                print(f"⚠️ No label found for {img_src}")

def create_data_yaml():
    data = {
        "path": BASE_DIR,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES
    }

    yaml_path = os.path.join(BASE_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data, f)

    print(f"📄 data.yaml created at: {yaml_path}")

if __name__ == "__main__":
    make_dirs()
    split_and_move_data()
    create_data_yaml()
    print("🎉 Dataset preparation complete!")
