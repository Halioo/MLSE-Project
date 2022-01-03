import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Variables liées à l'environnement

# seed utilisée pour garantir la reproductibilité des résultats
SEED = 42

# Fréquence d'échantillonnage de la carte
SENSORS_SAMPLING_RATE = 50 # Hz

# Chemin d'accès aux données
DATA_PATH = 'data/'

# fonctions utilitaires

def flatten(list):
    return [item for sublist in list for item in sublist]


# Chargement des données

from os import listdir
from os.path import isfile, join

dataset = []

csvs = [f for f in listdir(DATA_PATH) if isfile(join(DATA_PATH, f))]
for filename in csvs:
    dataset.append(pd.read_csv(DATA_PATH + filename))

raw_dataset = dataset.copy()
len(raw_dataset)


# Définition des limites de chaque colonne

# Limites des données utiles de chaque fichier .csv
LIMITES = [ [60, 650],   [50, 350],  [100, 600],
            [0, 550],    [0, 550],   [150, 250],
            [0, 700],    [75, 820],  [0, 500],
            [50, 550],   [50, 750],  [50, 900],
            [0, 900],    [50, 1000], [100, 1200],
            [50, 1300],  [50, 900],  [200, 700],
            [0, 60],     [0, 60],    [0, 25],
            [100, 800],  [100, 900], [100, 600],
            [100, 850],  [50, 900],  [100, 1500],
            [100, 2200], [30, 300],  [0, 650],
            [0, 550],    [100, 500], [0, 710]]

def trimDataset(ds):
    for i in range(len(ds)):
        ds[i] = ds[i][LIMITES[i][0]:LIMITES[i][1]]

trimDataset(dataset)

trimed_dataset = dataset.copy()


# Echantillonnage des données

DATA_SAMPLING_RATE = 10 # Hz

def sampleDf(df, scaleFactor):
    res = []
    for i in range(scaleFactor):
        res.append(df.iloc[lambda x: x.index % scaleFactor == i])
    return res

scalingFactor = SENSORS_SAMPLING_RATE // DATA_SAMPLING_RATE

for i in range(len(dataset)):
    dataset[i] = sampleDf(dataset[i], scalingFactor)

sampled_dataset = flatten(dataset.copy())
len(sampled_dataset)


# Section des fenêtres de capture

WINDOW_TIME = 2 # s

WINDOW_LENGTH = WINDOW_TIME * DATA_SAMPLING_RATE

def sliceDf(df, step):
    res = []
    while (len(df) > step):
        res.append(df.iloc[:step])
        df = df.iloc[1:]
    return res

for i in range(len(dataset)):
    for ii in range(len(dataset[i])):
        dataset[i][ii] = sliceDf(dataset[i][ii], WINDOW_LENGTH)
    dataset[i] = flatten(dataset[i])

sliced_dataset = flatten(dataset.copy())
len(sliced_dataset)

# Vérification de la données
# Toutes les dataframes font-elles bien la même taille ?

sizes = []
for df in sliced_dataset:
    if len(df) not in sizes:
        sizes.append(len(df))
sizes


# Elimination des colonnes inutiles

UNUSED_DATA_COLUMN = ["T [ms]"]

def cleanDf(df):
    return df.drop(columns=UNUSED_DATA_COLUMN)

print(dataset[0][0].columns)

for i in range(len(dataset)):
    for ii in range(len(dataset[i])):
        dataset[i][ii] = cleanDf(dataset[i][ii])

print(dataset[0][0].columns)

# Labellisation

from gSheet import SheetAPI

# The ID and range of a sample spreadsheet.
SPREADSHEET_ID = '1By59dQ56zL_kP0tW9Iyf4FppyEvJtcwnuG4gx1iokpM'
api = SheetAPI(SPREADSHEET_ID)
api.connect()

Y = api.getValues("A2:D100")
Ycolumns = Y[0]
Y = Y[0:]

for i in range(len(Y)):
    Y[i] = [Y[i] for _ in dataset[i]]

size_dataset = 0
size_Y = 0

for i in range(len(dataset)):
    size_dataset += len(dataset[i])
    size_Y += len(Y[i])
    assert(len(Y[i]) == len(dataset[i]))

print("size_dataset: ", size_dataset)
print("size_Y: ", size_Y)

Y = flatten(Y)
dataset = flatten(dataset)


# Création des données de test

from sklearn.model_selection import train_test_split

E_train, E_test, Y_train, Y_test = train_test_split(dataset, Y, test_size=0.33, random_state=SEED)

E_train = np.array(E_train)
Y_train = np.array(Y_train)
E_test = np.array(E_test)
Y_test = np.array(Y_test)

print(len(E_train) + len(E_test))


# Génération du modèle

from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model  # install graphviz on OS

from tensorflow.keras.optimizers import Adam

model = Sequential() # Instanciation du modèle

model.add(Dense(6, input_dim=20, activation='sigmoid')) # Ajout de la couche intermédiaire
model.add(Dense(2, activation='sigmoid')) # Ajout de la couche de sortie

# l'optimiseur ADAM : Adaptive Moment Estimation
# optimiseur très utilisé car efficace et consomme peu de mémoire
# on a une assez grande quantité de données pour entrainer ce réseau de neurones
opt = Adam()

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

import timeit

start_time = timeit.default_timer()

history = model.fit(E_train, Y_train, validation_split=0.15, shuffle=False, epochs=400, verbose=0, batch_size=5)

print("Temps passé : %.2fs" % (timeit.default_timer() - start_time))