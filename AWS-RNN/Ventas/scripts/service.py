#####################################
# -*- coding: utf-8 -*-
# Ernesto Garcia Barriga
# Javier Vega Garcia
# Recurrent Neural Network - RNN
#####################################
# REALIZAR PREDICCIONES - Final
#####################################
from flask import Flask, request
from executor import Modelo_RNA
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
import numpy  as np
import pandas as pd
import math

# Initialize the application service
app = Flask(__name__) # create the Flask app
global loaded_model, graph
loaded_model, graph = Modelo_RNA()

#Define a route
@app.route('/')
def main_page():
	return 'Bienvenido a RNN Enterprises !!!'

@app.route('/Ventas/', methods=['GET','POST'])
def ventas():
	return 'Modelo RNN de Ventas !!!'

@app.route('/Ventas/default/', methods=['GET','POST'])
def default():
#def index(name='Treehouse'):  # provide default value for 'name'
	#file_test = request.args.get('name', name) #<-- get argument 'name' or use default name set above
	file_test = request.args.get('name') # if key doesn't exist, returns None
	if len(file_test) == 0:
		return("Ingrese archivo correcto !!!")
	else:
		print(request.args)

	file_test = '../samples/' + file_test
	campo_sel = "Monto" # Campo desde donde se va a realizar la prediccion
	tam_rango = 13      # Tamaño del Rango para la prediccion
	dir_path = "../model/rnn_"
	file_bases = dir_path + "model.csv"
	file_scale = dir_path + "scale.save"

	# Obtener el Train
	df = pd.read_csv(file_bases)
	df_train = pd.DataFrame(df[campo_sel])
	nro_reg_train = df_train.shape[0]

	# Obtener el Test
	df = pd.read_csv(file_test)
	df_test = pd.DataFrame(np.log(df[campo_sel]))
	nro_reg_test = df_test.shape[0]
	sc = joblib.load(file_scale) # Normalizando campo de Calculo
	df_calculo = pd.DataFrame(sc.transform(df_test), columns = [campo_sel])

	# Obtener el Total
	df_total = pd.DataFrame(pd.concat((df_train[campo_sel], df_calculo[campo_sel]), axis = 0))

	# Obtener el Temporal de Calculo
	df_calculo = df_total[:].values
	df_calculo = df_calculo.reshape(-1,1) # Redimensionando

	# Transformando el Dataframe para Predecir
	X_test = []
	for i in range(tam_rango, tam_rango+nro_reg_test):
		X_test.append(df_calculo[i-tam_rango:i, 0])
	X_test = np.array(X_test)
	X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) # Redimensionando

	with graph.as_default():
		Y_test = loaded_model.predict(X_test)
		Y_test = sc.inverse_transform(Y_test) # Invirtiendo la  Normalizacion

		rmse = math.sqrt(mean_squared_error(df_test, Y_test))
		# Mostrando Resultados
		print("{0}Reg.{1}Original{2}Predicho".format(' '*5, ' '*10, ' '*8))
		print('{0} {1} {1}'.format('-'*13, '-'*16))
		for i in range(0, nro_reg_test):
			print('{}'.format(' '*4), '{:>04}'.format(i+1), '{}'.format(' '*2), '{:>16,.4f}'.format(np.exp(df_test[campo_sel][i])), '{}'.format(' '*0), '{:>15,.4f}'.format(np.exp(Y_test[i][0])))
		resultado = '\nEl Error Cuadratico medio es de: ' + '{:>,.8f}'.format(rmse)
		return(resultado)

# Run de application
app.run(host='0.0.0.0', port=5000) # run app in host on port 5000
