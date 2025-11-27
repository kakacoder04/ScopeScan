import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GaussianNoise, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image

train_set = 'input/train'
val_set = 'input/val'
test_set = 'input/test'

train_datagen = image.ImageDataGenerator(
    rescale=1./255,  
    rotation_range=15,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

validation_datagen = image.ImageDataGenerator(rescale=1./255)
test_datagen = image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_set,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    val_set,
    target_size=(224, 224),
    batch_size=16,
    shuffle=False, 
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_set,
    target_size=(224, 224),
    batch_size=16,
    shuffle=False,
    class_mode='categorical'
)

print("Class indices:", train_generator.class_indices)
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")

base_model = tf.keras.applications.EfficientNetB2(
    weights='imagenet', 
    input_shape=(224, 224, 3), 
    include_top=False
)

for layer in base_model.layers:
    layer.trainable = False

model = Sequential([
    base_model,
    GaussianNoise(0.35),
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    GaussianNoise(0.35),
    Dropout(0.2),
    Dense(4, activation='softmax')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy', 
    metrics=['accuracy', 'Precision', 'Recall', 'AUC']
)

steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = validation_generator.samples // validation_generator.batch_size

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=1,
    validation_data=validation_generator,
    validation_steps=validation_steps
)

print("\n=== Training Set Evaluation ===")
train_results = model.evaluate(train_generator)
print("\n=== Validation Set Evaluation ===")
val_results = model.evaluate(validation_generator)
print("\n=== Test Set Evaluation ===")
test_results = model.evaluate(test_generator)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Accuracy and Loss")
plt.plot(history.history["accuracy"], label="train_accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
best_epoch = np.argmin(history.history["val_loss"])
plt.plot(best_epoch, history.history["val_loss"][best_epoch], 
         marker="x", markersize=10, color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.title("Precision")
plt.plot(history.history["precision"], label="train_precision")
plt.plot(history.history["val_precision"], label="val_precision")
best_precision_epoch = np.argmax(history.history["val_precision"])
plt.plot(best_precision_epoch, history.history["val_precision"][best_precision_epoch], 
         marker="x", markersize=10, color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

model.save('model_test.h5')
print("\nModel saved as 'model_test.h5'")