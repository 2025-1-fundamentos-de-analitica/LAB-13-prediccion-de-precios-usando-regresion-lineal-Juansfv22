#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

# Este script construye un modelo para predecir el precio actual de vehículos usados.
# El dataset original ya está separado en entrenamiento y prueba y se encuentra en "files/input/".

import pandas as pd
import os
import json
import gzip
import pickle

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error

# Paso 1: Limpieza de datos. Agregamos columna "Age" y eliminamos columnas irrelevantes.
def preparar_datos(ruta_csv):
    datos = pd.read_csv(ruta_csv, compression='zip', index_col=False)
    datos["Age"] = 2021 - datos["Year"]
    datos.drop(columns=["Year", "Car_Name"], inplace=True)
    return datos

ruta_entrenamiento = 'files/input/train_data.csv.zip'
ruta_prueba = 'files/input/test_data.csv.zip'

datos_entrenamiento = preparar_datos(ruta_entrenamiento)
datos_prueba = preparar_datos(ruta_prueba)

# Paso 2: División de características y variable objetivo.
X_train = datos_entrenamiento.drop(columns=["Present_Price"])
y_train = datos_entrenamiento["Present_Price"]

X_test = datos_prueba.drop(columns=["Present_Price"])
y_test = datos_prueba["Present_Price"]

# Paso 3: Definición del pipeline del modelo.
categoricas = ['Fuel_Type', 'Selling_type', 'Transmission']
numericas = [col for col in X_train.columns if col not in categoricas]

transformador = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categoricas),
        ('num', MinMaxScaler(), numericas)
    ],
    remainder='passthrough'
)

pipeline_regresion = Pipeline(steps=[
    ('transformador', transformador),
    ('k_best', SelectKBest(f_regression)),
    ('modelo', LinearRegression())
])

# Paso 4: Optimización del modelo usando GridSearchCV.
parametros = {
    'k_best__k': [5, 10, 15, 20],
    'modelo__fit_intercept': [True, False]
}

busqueda = GridSearchCV(
    estimator=pipeline_regresion,
    param_grid=parametros,
    cv=10,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)

busqueda.fit(X_train, y_train)

# Paso 5: Guardado del modelo en formato gzip.
os.makedirs("files/models", exist_ok=True)
ruta_modelo = 'files/models/model.pkl.gz'

with gzip.open(ruta_modelo, 'wb') as salida:
    pickle.dump(busqueda, salida)

# Paso 6: Evaluación del modelo y exportación de métricas.
def exportar_metricas(modelo, xtr, xts, ytr, yts):
    pred_tr = modelo.predict(xtr)
    pred_ts = modelo.predict(xts)

    resultados = [
        {
            "type": "metrics",
            "dataset": "train",
            "r2": r2_score(ytr, pred_tr),
            "mse": mean_squared_error(ytr, pred_tr),
            "mad": median_absolute_error(ytr, pred_tr)
        },
        {
            "type": "metrics",
            "dataset": "test",
            "r2": r2_score(yts, pred_ts),
            "mse": mean_squared_error(yts, pred_ts),
            "mad": median_absolute_error(yts, pred_ts)
        }
    ]

    os.makedirs("files/output", exist_ok=True)
    with open("files/output/metrics.json", "w") as f:
        for entrada in resultados:
            f.write(json.dumps(entrada) + "\n")

exportar_metricas(busqueda, X_train, X_test, y_train, y_test)
