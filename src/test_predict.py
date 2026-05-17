import tensorflow as tf
import numpy as np
import os
import random

# configuration
MODEL_PATH = 'models/best_pro_model.keras'
TEST_DIR = 'final_dataset/Testing'
IMG_SIZE = (224, 224)
CLASSES = ['glioma', 'meningioma', 'normal', 'pituitary']
THRESHOLD = 0.85 

print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

def test_random_samples(samples_per_folder=10):
    overall_results = []

    print(f"\nStarting Random Batch Test ({samples_per_folder} per class)")

    for category in CLASSES:
        category_path = os.path.join(TEST_DIR, category)
        if not os.path.exists(category_path):
            print(f"Warning: Folder {category} not found.")
            continue

        # 10 random files
        all_images = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        sample_files = random.sample(all_images, min(len(all_images), samples_per_folder))

        print(f"\nTesting Category: {category.upper()}")
        print("-" * 30)

        for filename in sample_files:
            img_path = os.path.join(category_path, filename)
            
            # preprocessing
            img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            # prediction
            preds = model.predict(img_array, verbose=0)
            score = preds[0]
            class_idx = np.argmax(score)
            confidence = score[class_idx]
            pred_label = CLASSES[class_idx]

            is_correct = (pred_label == category)
            overall_results.append(is_correct)

            status = "✓ CORRECT" if is_correct else "✗ WRONG"
            conf_str = f"{confidence*100:.2f}%"
            
            if confidence < THRESHOLD:
                print(f"[{status}] {filename}: {pred_label} ({conf_str}) - ! LOW CONFIDENCE")
            else:
                print(f"[{status}] {filename}: {pred_label} ({conf_str})")

    # summary
    total = len(overall_results)
    correct = sum(overall_results)
    print(f"\n" + "="*30)
    print(f"BATCH TEST SUMMARY")
    print(f"Total Tested: {total}")
    print(f"Total Correct: {correct}")
    print(f"Batch Accuracy: {(correct/total)*100:.2f}%")
    print("="*30)

if __name__ == "__main__":
    test_random_samples(10)