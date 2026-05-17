import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import os
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = 'final_dataset' 
TRAIN_DIR = os.path.join(DATA_DIR, 'Training')
TEST_DIR = os.path.join(DATA_DIR, 'Testing')
MODELS_DIR = 'models'
IMG_SIZE = (224, 224) 
BATCH_SIZE = 32
EPOCHS = 50 
CLASSES = ['glioma', 'meningioma', 'normal', 'pituitary']

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)


print("Loading and Splitting Datasets...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True,
    validation_split=0.2,
    subset="training",
    seed=42
).prefetch(buffer_size=tf.data.AUTOTUNE)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True,
    validation_split=0.2,
    subset="validation",
    seed=42
).prefetch(buffer_size=tf.data.AUTOTUNE)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, 
    image_size=IMG_SIZE, 
    batch_size=BATCH_SIZE, 
    label_mode='categorical', 
    shuffle=False
).prefetch(buffer_size=tf.data.AUTOTUNE)


def build_model():
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(224, 224, 3), 
        include_top=False, 
        weights='imagenet'
    )
    
    base_model.trainable = True
    for layer in base_model.layers[:-40]: 
        layer.trainable = False

    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        
        # intensity jitter augmentations (destroys background artifacts)
        layers.RandomBrightness(factor=0.2, value_range=(0.0, 255.0)),
        layers.RandomContrast(factor=0.2),
        
        # geometric transformations
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.3),
        layers.RandomZoom(0.3),
        
        base_model,
        
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(512, activation='relu'), 
        layers.BatchNormalization(),
        layers.Dropout(0.5), 
        
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        
        layers.Dense(4, activation='softmax')
    ])
    return model

model = build_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

custom_class_weights = {
    0: 2.0,  
    1: 1.5,  
    2: 1.0, 
    3: 1.0
}

checkpoint = callbacks.ModelCheckpoint(
    filepath=os.path.join(MODELS_DIR, "best_pro_model.keras"),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

lr_reducer = callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=3, 
    min_lr=1e-7,
    verbose=1
)

early_stop = callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True,
    verbose=1
)

print("\nStarting Training Process")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=custom_class_weights, 
    callbacks=[early_stop, checkpoint, lr_reducer]
)

print("\nFinal Test Set Evaluation")
test_loss, test_acc = model.evaluate(test_ds)
print(f"\nFINAL TEST ACCURACY: {test_acc*100:.2f}%")

y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_true_indices = np.argmax(y_true, axis=1)
predictions = model.predict(test_ds)
y_pred_indices = np.argmax(predictions, axis=1)

print("\nClassification Report:")
print(classification_report(y_true_indices, y_pred_indices, target_names=CLASSES))