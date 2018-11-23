#####################################
# -*- coding: utf-8 -*-
# Ernesto Garcia Barriga
# Javier Vega Garcia
# Recurrent Neural Network - RNN
#####################################
# ACTIVAR EL SERVICIO - Final
#####################################
import tensorflow as tf
#import matplotlib.pyplot as plt
from keras.models import model_from_json

def Modelo_RNA():
	print("Cargando modelo desde el disco ...")
	dir_path = "../model/rnn_"
	file_model = dir_path + "model.json"
	file_pesos = dir_path + "model.h5"

	json_file = open(file_model, 'r')
	loaded_model_json = json_file.read()
	json_file.close()

	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights(file_pesos)
	loaded_model.compile(optimizer = 'adam', loss = 'mean_squared_error')

	print("Modelo cargado !!!")
	graph = tf.get_default_graph()
	return loaded_model, graph