from tensorflow.keras.applications import EfficientNetB0 # type: ignore
import opendatasets as od # type: ignore
import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore
import kagglehub
from kagglehub import KaggleDatasetAdapter
from kaggle.api.kaggle_api_extended import KaggleApi # type: ignore
from tensorflow.keras.preprocessing import image_dataset_from_directory # type: ignore
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras import layers, models, regularizers # type: ignore
import matplotlib.pyplot as plt
from pathlib import Path

# dataset_link="https://www.kaggle.com/datasets/nikolasgegenava/sneakers-classification"
# od.download(dataset_link)

data_dir = Path(__file__).parent.parent.parent / 'sneakers-dataset' / 'sneakers-dataset'
BATCH_SIZE = 32 # Tamaño del lote para entrenamiento
IMG_SIZE = (224, 224) # Tamaño de imagen para el modelo
EPOCHS = 15 # Número de épocas para entrenamiento

base_model = EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)
base_model.trainable = False

train_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123, 
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

model = tf.keras.Sequential([
    base_model,
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(50, activation='softmax')  # 50 clases
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    epochs=14,
    batch_size= BATCH_SIZE,
)

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='#1f77b4', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#ff7f0e', linewidth=2, linestyle='--')
plt.title('Accuracy durante el Entrenamiento', fontsize=12)
plt.xlabel('Épocas', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.ylim([0, 1])
plt.xticks(np.arange(0, 14, step=2))
plt.grid(True, alpha=0.3)
plt.legend()


model.save('predictSneakersModelPreTrainned.h5')