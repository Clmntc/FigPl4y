# Déploiement streamlit
import streamlit as st
import cv2
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import numpy as np
# Visualisation
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
# Sklearn metrics 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix, classification_report
# Holdout
from sklearn.model_selection import train_test_split
# Pickle
import pickle
from PIL import Image
# Keras 
# from tensorflow import keras
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation,Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from tensorflow.keras.optimizers import Adam
# from keras.models import load_model

# st.markdown(
#     """
#     <style>
#     .reportview-container {
#         background: 'white'
#     }
#    .sidebar .sidebar-content {
#         background: 'white'
#     }
#     </style>
#     """,unsafe_allow_html=True
# )

# image = Image.open('C:/Users/Utilisateur/Desktop/Alyson/Flasks/Openfood/NP.png')
# st.image(image)

##########################################
##### Import du dataset test et du modèle
@st.cache(allow_output_mutation=True)
def load_pred():
    model = load_model('C:/Users/Utilisateur/Desktop/Arturo/Reseau_neuronal_convolutif/Amodel.h5')
    return model

model = load_pred()

####################
##### TEST ALEATOIRE 

st.title("Tu peux pas TEST !")

# SIZE = 256

# if 'key' not in st.session_state:
#     st.session_state.key = 0

# def load_jeu():
#     canvas_result = st_canvas(
#         fill_color='#000000',
#         stroke_width=20,
#         stroke_color='#FFFFFF',
#         background_color='#000000',
#         width=SIZE,
#         height=SIZE,
#         # display_toolbar='reset',
#         drawing_mode="freedraw",
#         key=f'canvas{st.session_state.key}'
#         )

#     if canvas_result.image_data is not None:
#         img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))

#     model = load_model('C:/Users/Utilisateur/Desktop/Arturo/Reseau_neuronal_convolutif/my_model.h5')

#     if st.button('Predict'):
#         test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         val = model.predict(test_x.reshape(1, 28, 28, 1))
#         st.write(f'result: {np.argmax(val[0])}')

# load_jeu()


# if st.button('Nouvel essai'):
#     st.session_state.key += 1
#     st.write(load_jeu())



# ##################
# ##### Statistiques

# st.title('Est-ce que le resultat est bon ?')

# # Instancier le décompte
# if 'total' not in st.session_state:
#     st.session_state.total = 0
# if 'oui' not in st.session_state:
#     st.session_state.oui = 0

# # Les boutons
# if st.button('oui'):
#     st.session_state.oui += 1
#     st.session_state.total += 1
# if st.button('non'):
#     st.session_state.total += 1

# total = st.session_state.total
# oui = st.session_state.oui

# try:
#     st.write('Count = ', oui/total*100)
# except:
#   pass

######################
##### DESSIN ALEATOIRE 


# streamlit run C:/Users/Utilisateur/Desktop/Arturo/Reseau_neuronal_convolutif/Streamlit/FigPlay.py --server.maxUploadSize=1028


# # créer environnement python pour l'app
# python -m venv myenv

# # activer l'environnement python
# # allez dans le dossier myenv avec 
# cd venv
# # puis 
# ./Scripts/activate
