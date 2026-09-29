import tensorflow as tf
from tensorflow.keras import layers, models

# Dataset path
DATASET_PATH = r"C:\PlantVillage\Plant_leave_diseases_dataset_without_augmentation"

# Settings
IMG_SIZE = (96, 96)
BATCH_SIZE = 32
EPOCHS = 5

# Load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Number of classes:", num_classes)
print("Classes:", class_names)

# Improve performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# CNN Model
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(96, 96, 3)),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(num_classes, activation="softmax")
])

# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
print("\nStarting model training...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Save model
model.save("model/plant_disease_model.keras")

# Save class names
with open("model/class_names.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

print("\n================================")
print("MODEL TRAINING COMPLETED!")
print("Model saved successfully.")
print("================================")