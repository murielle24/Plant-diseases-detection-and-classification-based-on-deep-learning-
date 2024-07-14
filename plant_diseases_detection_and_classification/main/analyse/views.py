from django.shortcuts import render

# Create your views here.




# page d'acceuil 
def index(request):
    return render(request, 'index.html')




# page d'acceuil 
def resultats(request):
    return render(request, 'resultats.html')




# views.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings

# Définir les fonctions de métrique personnalisées (si nécessaire)
def precision_m(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision

def recall_m(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

# Charger le modèle
model_inception = tf.keras.models.load_model(os.path.join(settings.BASE_DIR, 'Inception.h5'), custom_objects={'precision_m': precision_m, 'recall_m': recall_m, 'f1_m': f1_m})

model_alexnet = tf.keras.models.load_model(os.path.join(settings.BASE_DIR, 'AlexNet.h5'), custom_objects={'precision_m': precision_m, 'recall_m': recall_m, 'f1_m': f1_m})

# Dictionnaire des labels de classe
class_labels = {"0": "Apple___Apple_scab", "1": "Apple___Black_rot", "2": "Apple___Cedar_apple_rust", "3": "Apple___healthy", "4": "Blueberry___healthy", "5": "Cherry_(including_sour)___Powdery_mildew", "6": "Cherry_(including_sour)___healthy", "7": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "8": "Corn_(maize)___Common_rust_", "9": "Corn_(maize)___Northern_Leaf_Blight", "10": "Corn_(maize)___healthy", "11": "Grape___Black_rot", "12": "Grape___Esca_(Black_Measles)", "13": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "14": "Grape___healthy", "15": "Orange___Haunglongbing_(Citrus_greening)", "16": "Peach___Bacterial_spot", "17": "Peach___healthy", "18": "Pepper,_bell___Bacterial_spot", "19": "Pepper,_bell___healthy", "20": "Potato___Early_blight", "21": "Potato___Late_blight", "22": "Potato___healthy", "23": "Raspberry___healthy", "24": "Soybean___healthy", "25": "Squash___Powdery_mildew", "26": "Strawberry___Leaf_scorch", "27": "Strawberry___healthy", "28": "Tomato___Bacterial_spot", "29": "Tomato___Early_blight", "30": "Tomato___Late_blight", "31": "Tomato___Leaf_Mold", "32": "Tomato___Septoria_leaf_spot", "33": "Tomato___Spider_mites Two-spotted_spider_mite", "34": "Tomato___Target_Spot", "35": "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "36": "Tomato___Tomato_mosaic_virus", "37": "Tomato___healthy"}

def upload_image(request):
    predicted_class = None
    if request.method == 'POST' and request.FILES.get('file'):
        # Sauvegarder le fichier téléchargé temporairement
        file = request.FILES['file']
        file_path = default_storage.save(file.name, file)

        # Charger l'image
        img_path = os.path.join(settings.MEDIA_ROOT, file_path)
        img = image.load_img(img_path, target_size=(224, 224))

        # Prétraiter l'image
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Faire la prédiction avec inception
        predictions = model_inception.predict(img_array)
        predicted_class_indices = np.argmax(predictions[0][0], axis=-1)
        incep_predicted_class = class_labels[str(predicted_class_indices)]

        # Charger l'image
        img_path = os.path.join(settings.MEDIA_ROOT, file_path)
        img = image.load_img(img_path, target_size=(227, 227))

        # Prétraiter l'image
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        # Faire la prédiction avec alexnet
        predictions = model_alexnet.predict(img_array)
        predicted_class_indices = np.argmax(predictions[0][0], axis=-1)
        alex_predicted_class = class_labels[str(predicted_class_indices)]

        # Supprimer le fichier temporaire
        # default_storage.delete(file_path)
        # print({'predicted_class': predicted_class})

    return render(
        request,
        'resultats.html',
        {
            'incep_predicted_class': incep_predicted_class,
            'alex_predicted_class': alex_predicted_class,
            'path_img':"/media/"+file_path,
            'file':file
        }
    )
