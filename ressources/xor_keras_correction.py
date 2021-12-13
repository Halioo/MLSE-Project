import timeit

import matplotlib.pyplot as plt
import numpy as np
from keras import optimizers
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import plot_model  # install graphviz on OS
from sklearn.model_selection import train_test_split
# fix random seed for reproducibility
from tensorflow.python.framework.random_seed import set_random_seed

seed = 0
np.random.seed(seed)
set_random_seed(seed)

# Définition des exemples entrées / sorties
# Couples de valeurs en entrée :
# [0,0], [0.1,0], [0.2,0], ... [0.9,0], [1,0], [0,0.1], [0.1,0.1], ...
E1 = np.repeat(np.linspace(0, 1, 10).reshape([1, 10]), 10, axis=0)
E1 = E1.reshape(100)
E2 = np.repeat(np.linspace(0, 1, 10), 10)
E = np.array([E1, E2])

# Sorties correspondantes
# 0, 0, 0, ... 1, 1, 0, 0, ...
Y = np.zeros(E.shape[1])
for n_exe in range(0, E.shape[1]):
    if (E[0, n_exe] >= 0.5 > E[1, n_exe]) or (E[0, n_exe] < 0.5 <= E[1, n_exe]):
        Y[n_exe] = 1

# plt.figure()
# plt.plot(E[0,:])
# plt.plot(E[1,:])
# plt.plot(Y)
# plt.legend(('E1','E2','Y'))

# Data cleanning, need to transpose E
Et=E.transpose()
E_train, E_test, Y_train, Y_test = train_test_split(Et, Y, test_size=0.33, random_state=seed)

# create model
model = Sequential()
model.add(Dense(3, input_dim=2, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
opt = optimizers.SGD(lr=0.9, decay=0, momentum=0.2)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['acc'])
# print model in .png file
#plot_model(model)

#train
start_time = timeit.default_timer()
history = model.fit(E_train, Y_train, validation_split=0.15, shuffle=False, epochs=400, verbose=0, batch_size=5)
print("Temps passé : %.2fs" % (timeit.default_timer() - start_time))

#plot figure
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
# plt.show()
# Plot training & validation loss values

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
# plt.show()

# evaluate the model
scores = model.evaluate(E_test, Y_test)
print("\nEvaluation sur le test data %s: %.2f - %s: %.2f%% " % (
    model.metrics_names[0], scores[0], model.metrics_names[1], scores[1] * 100))

#evaluate with the all dataset and plot
prediction = model.predict_on_batch(Et)
prediction = prediction.reshape(10, 10)
attendues = Y.reshape(10, 10)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(attendues, extent=[0, 1, 0, 1])
plt.title('Cartopgrahie de la fonction attendue')
plt.xlabel('Entree 1')
plt.subplot(1, 2, 2)
plt.imshow(prediction, extent=[0, 1, 0, 1])
plt.title('Cartopgrahie de la fonction predite')
plt.xlabel('Entree 1')
plt.ylabel('Entree 2')
plt.show()
