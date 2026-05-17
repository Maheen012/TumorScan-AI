import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = 'final_dataset' 
TEST_DIR = os.path.join(DATA_DIR, 'Testing')
MODEL_PATH = "models/best_pro_model.keras"
CLASSES = ['glioma', 'meningioma', 'normal', 'pituitary']

model = tf.keras.models.load_model(MODEL_PATH)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, image_size=(224, 224), batch_size=32, label_mode='categorical', shuffle=False
)

y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_true_indices = np.argmax(y_true, axis=1)
predictions = model.predict(test_ds)
y_pred_indices = np.argmax(predictions, axis=1)

cm = confusion_matrix(y_true_indices, y_pred_indices)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Final Confusion Matrix (Unseen Test Data)')
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')

plt.savefig('screenshots/confusion_matrix.png', bbox_inches='tight', dpi=300)
plt.show() 