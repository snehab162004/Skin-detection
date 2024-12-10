import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing import image
from preprocessed import load_data  # Import load_data function from the preprocessed file

# Building the model
def build_model(img_size=128, num_classes=6):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Load datasets
train_ds = load_data("./processed_dataset/train")
val_ds = load_data("./processed_dataset/val")
test_ds = load_data("./processed_dataset/test")

# Create the model
model = build_model(num_classes=len(os.listdir("./processed_dataset/train")))

# Step 3: Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# Step 4: Evaluate the model
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test Accuracy: {test_accuracy}")

# Step 5: Visualize training results
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



# Step 7: Save the model
model.save("skin_disease_model.h5")
print("Model saved successfully!")
