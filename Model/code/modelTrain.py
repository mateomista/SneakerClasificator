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
    seed=123,  # Para reproducibilidad
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

print("Número de imágenes de entrenamiento:", train_ds.samples)
print("Número de imágenes de validación:", val_ds.samples)
print("Clases encontradas:", train_ds.class_indices)