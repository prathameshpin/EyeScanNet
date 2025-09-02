# Importing required libraries
import matplotlib.pyplot as plt
from tensorflow.keras import layers # type: ignore
from tensorflow.keras import Sequential # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.utils import image_dataset_from_directory # type: ignore
from tensorflow.keras.applications import VGG19 # type: ignore

# Constants
path = "dataset"
batch_size = 32
img_size = (224,224)
epochs = 12
categories = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]
random_seed = 617
val_split = 0.2

# Getting training data from given directory (80%)
train_data = image_dataset_from_directory(
    path,
    validation_split = val_split,
    subset = "training",
    seed = random_seed,
    image_size = img_size,
    batch_size = batch_size,
    label_mode = 'categorical'
)

# Getting validation data from given directory (20%)
val_data = image_dataset_from_directory(
    path,
    validation_split = val_split,
    subset = "validation",
    seed = random_seed,
    image_size = img_size,
    batch_size = batch_size,
    label_mode = 'categorical'
)

# Scaling all pixels between 0 and 1
normalization_layer = layers.Rescaling(1./255)
train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
val_data = val_data.map(lambda x, y: (normalization_layer(x), y))

# Augmenting some training data to make model more robust
augmentation_layer = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])
train_data = train_data.map(lambda x, y: (augmentation_layer(x, training=True), y))

# Get a base model and freeze its pretrained layers
base_model = VGG19(weights = "imagenet", input_shape = (224,224,3), include_top = False)
for layer in base_model.layers:
    layer.trainable = False

# Build a new model on top of base model
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(32, activation="relu")(x)
x = layers.Dropout(0.1)(x)
output = layers.Dense(len(categories), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(train_data, validation_data=val_data, epochs=epochs)

# Model accuracy plots
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Model Accuracy")
plt.show()

# Saving the model for further predictions
model.save("model.keras")