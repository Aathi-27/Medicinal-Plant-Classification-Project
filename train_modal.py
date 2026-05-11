import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# ==========================================
# DATASET PATH
# ==========================================

dataset_path = "augmented_dataset"

# ==========================================
# IMAGE SETTINGS
# ==========================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# ==========================================
# DATA GENERATOR
# ==========================================

datagen = ImageDataGenerator(
    validation_split=0.2,
    rescale=1./255
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# ==========================================
# CONVNEXT WRAPPER
# ==========================================

def ConvNeXtModel():

    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224,224,3)
    )

    base_model.trainable = False

    model = Sequential([

        base_model,

        GlobalAveragePooling2D(),

        Dense(256, activation='relu'),

        Dropout(0.3),

        Dense(train_data.num_classes, activation='softmax')

    ])

    return model

# ==========================================
# BUILD MODEL
# ==========================================

model = ConvNeXtModel()

# ==========================================
# COMPILE MODEL
# ==========================================

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# TRAIN MODEL
# ==========================================

history = model.fit(

    train_data,

    validation_data=val_data,

    epochs=5

)

# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs("model", exist_ok=True)

model.save("model/plant_model.h5")

print("\nMODEL SAVED SUCCESSFULLY")

# ==========================================
# ACCURACY GRAPH
# ==========================================

plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title("Training vs Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend(["Train", "Validation"])

plt.savefig("accuracy_graph.png")

plt.show()

# ==========================================
# LOSS GRAPH
# ==========================================

plt.figure(figsize=(8,5))

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title("Training vs Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend(["Train", "Validation"])

plt.savefig("loss_graph.png")

plt.show()

print("\nTRAINING COMPLETED")