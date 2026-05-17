import os
import shutil
import random
import imagehash
from PIL import Image
from tqdm import tqdm

RAW_DATA_SRC = "raw_data" 

FINAL_DATASET_DIR = "final_dataset"
QUARANTINE_DIR = "duplicates"

CLASSES = ['glioma', 'meningioma', 'normal', 'pituitary']

# Test split ratio 
TEST_SPLIT_RATIO = 0.2
SEED = 42
random.seed(SEED)

def build_pristine_dataset():
    print("Initializing Duplication Check...")
    
    unique_images_by_class = {c: [] for c in CLASSES}
    hash_registry = {}
    duplicate_counter = 0
    total_scanned = 0

    if not os.path.exists(QUARANTINE_DIR):
        os.makedirs(QUARANTINE_DIR)

    for root, dirs, files in os.walk(RAW_DATA_SRC):
        folder_name = os.path.basename(root).lower()
        if any(x in root for x in [QUARANTINE_DIR, FINAL_DATASET_DIR]):
            continue
            
        current_class = None
        for c in CLASSES:
            if c in folder_name:
                current_class = c
                break
                
        if not current_class:
            continue

        print(f"Analyzing raw files in folder: [{current_class.upper()}]")
        for file in tqdm(files, desc="Hashing structural layers"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                total_scanned += 1
                file_path = os.path.join(root, file)
                
                try:
                    with Image.open(file_path) as img:
                        img_hash = str(imagehash.phash(img))
                        
                    if img_hash in hash_registry:
                        duplicate_counter += 1
                        
                        dest_q_folder = os.path.join(QUARANTINE_DIR, current_class)
                        if not os.path.exists(dest_q_folder):
                            os.makedirs(dest_q_folder)
                        shutil.move(file_path, os.path.join(dest_q_folder, f"dup_{duplicate_counter}_{file}"))
                    else:
                        hash_registry[img_hash] = file_path
                        unique_images_by_class[current_class].append(file_path)
                        
                except Exception as e:
                    pass 

    print(f"\n Removal of duplicates Complete! Found {duplicate_counter} duplicates out of {total_scanned} files.")
    print("Generating Train/Test Dataset Splits...")

    train_root = os.path.join(FINAL_DATASET_DIR, "Training")
    test_root = os.path.join(FINAL_DATASET_DIR, "Testing")

    for class_name, file_list in unique_images_by_class.items():
        if not file_list:
            print(f"Warning: No images found for class {class_name}.")
            continue
            
        random.shuffle(file_list)
        
        split_idx = int(len(file_list) * (1 - TEST_SPLIT_RATIO))
        train_files = file_list[:split_idx]
        test_files = file_list[split_idx:]
        
        os.makedirs(os.path.join(train_root, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_root, class_name), exist_ok=True)

        for f in train_files:
            shutil.copy(f, os.path.join(train_root, class_name, os.path.basename(f)))
            
        for f in test_files:
            shutil.copy(f, os.path.join(test_root, class_name, os.path.basename(f)))

    print("complete!")
    
if __name__ == "__main__":
    build_pristine_dataset()