# -*- coding: utf-8 -*-
"""
Created on Sun May 28 10:20:36 2023

@author: cedri
"""

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import keras
import cv2
import pickle
from tensorflow.keras.utils import img_load, img_to_array
import keras


# Chargement du modèle
model = keras.models.load_model('model_cnn_VGG.pkl')

# Définition des classes de chiens
# Importing labels name
my_content = open("Dog_Class.txt", "r")
dog_names = my_content.read()
dogs_list = dog_names.split('\n')
my_content.close()    

# Chargement et prétraitement de l'image
def preprocess_image(image_path):
    

    img = Image.open(image_path)
    img = cv2.resize(img,(224,224))
    img = img.reshape(1,224,224,3)
    img = img.convert('RGB')
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalisation des valeurs de pixel
    img_array = np.expand_dims(img_array, axis=0)  # Ajout d'une dimension pour correspondre aux attentes du modèle
    return img_array

# Chargement de l'image à classer
image_path = input("Entrez le chemin vers l'image : ")
input_image = preprocess_image(image_path)

# Classification de l'image
predictions = model.predict(input_image)
predicted_class = np.argmax(predictions)

# Affichage du résultat
predicted_breed = my_content[predicted_class]
print("La classe prédite pour cette image est :", predicted_breed)
