import tensorflow as tf
import numpy as np
from array import array
import random

tau_features = np.loadtxt('tau_data/15GeV/15gev_tau_features.csv', delimiter=',')
tau_labels = np.loadtxt('tau_data/15GeV/15gev_tau_labels.csv', delimiter=',')
antitau_features = np.loadtxt('tau_data/15GeV/15gev_antitau_features.csv', delimiter=',')
antitau_labels = np.loadtxt('tau_data/15GeV/15gev_antitau_labels.csv', delimiter=',')

tau_features = np.concatenate((tau_features, np.array(np.loadtxt('tau_data/9.5GeV/9_5gev_tau_features.csv', delimiter=','))), axis=0)
tau_labels = np.concatenate((tau_labels, np.array(np.loadtxt('tau_data/9.5GeV/9_5gev_tau_labels.csv', delimiter=','))), axis = 0)
antitau_features = np.concatenate((antitau_features, np.array(np.loadtxt('tau_data/9.5GeV/9_5gev_antitau_features.csv', delimiter=','))), axis=0)
antitau_labels = np.concatenate((antitau_labels, np.array(np.loadtxt('tau_data/9.5GeV/9_5gev_antitau_labels.csv', delimiter=','))), axis=0)

tau_features = np.concatenate((tau_features, np.array(np.loadtxt('tau_data/5GeV/5gev_tau_features.csv', delimiter=','))), axis=0)
tau_labels = np.concatenate((tau_labels,np.array( np.loadtxt('tau_data/5GeV/5gev_tau_labels.csv', delimiter=','))), axis=0)
antitau_features = np.concatenate((antitau_features, np.array(np.loadtxt('tau_data/5GeV/5gev_antitau_features.csv', delimiter=','))), axis = 0)
antitau_labels = np.concatenate((antitau_labels, np.array(np.loadtxt('tau_data/5GeV/5gev_antitau_labels.csv', delimiter=','))), axis = 0)

indices = []
anti_indices = []
for i in range(len(tau_features)):
    indices.append(i)
indices = tf.random.shuffle(indices)
tau_features = tf.gather(tau_features, indices)
tau_labels = tf.gather(tau_labels, indices)
for i in range(len(antitau_features)):
    anti_indices.append(i)
antitau_features = tf.gather(antitau_features, anti_indices)
antitau_labels = tf.gather(antitau_labels, anti_indices)

tau_features_train = tau_features[0: int(0.9 * tau_features.shape[0]), :]
tau_features_test = tau_features[int(0.9 * tau_features.shape[0]):, :]
tau_labels_train = tau_labels[0: int(0.9 * tau_labels.shape[0]), :]
tau_labels_test = tau_labels[int(0.9 * tau_labels.shape[0]):, :]

antitau_features_train = antitau_features[0: int(0.9 * antitau_features.shape[0]), :]
antitau_features_test = antitau_features[int(0.9 * antitau_features.shape[0]):, :]
antitau_labels_train = antitau_labels[0: int(0.9 * antitau_labels.shape[0]), :]
antitau_labels_test = antitau_labels[int(0.9 * antitau_labels.shape[0]):, :]

#Add identifier feature, concatenate, shuffle. 1 = tau, 0 = anti
tau_identifier_features_train = np.ones((len(tau_features_train), 1))
tau_identifier_features_test = np.ones((len(tau_features_test), 1))
antitau_identifier_features_train = np.zeros((len(antitau_features_train), 1))
antitau_identifier_features_test = np.zeros((len(antitau_features_test), 1))

tau_identifier_labels_train = np.ones((len(tau_labels_train), 1))
tau_identifier_labels_test = np.ones((len(tau_labels_test), 1))
antitau_identifier_labels_train = np.zeros((len(antitau_labels_train), 1))
antitau_identifier_labels_test = np.zeros((len(antitau_labels_test), 1))

tau_features_train = np.append(tau_features_train, tau_identifier_features_train, axis=1)
tau_features_test = np.append(tau_features_test, tau_identifier_features_test, axis=1)
antitau_features_train = np.append(antitau_features_train, antitau_identifier_features_train, axis=1)
antitau_features_test = np.append(antitau_features_test, antitau_identifier_features_test, axis=1)

tau_labels_train = np.append(tau_labels_train, tau_identifier_labels_train, axis=1)
tau_labels_test = np.append(tau_labels_test, tau_identifier_labels_test, axis=1)
antitau_labels_train = np.append(antitau_labels_train, antitau_identifier_labels_train, axis=1)
antitau_labels_test = np.append(antitau_labels_test, antitau_identifier_labels_test, axis=1)

combined_features_train = np.concatenate((tau_features_train, antitau_features_train), axis=0)
combined_features_test = np.concatenate((tau_features_test, antitau_features_test), axis=0)
combined_labels_train = np.concatenate((tau_labels_train, antitau_labels_train), axis=0)
combined_labels_test = np.concatenate((tau_labels_test, antitau_labels_test), axis=0)

combined_features_labels_train = list(zip(combined_features_train, combined_labels_train))
random.shuffle(combined_features_labels_train)
combined_features_train, combined_labels_train = zip(*combined_features_labels_train)
combined_features_train = np.array(combined_features_train)
combined_labels_train = np.array(combined_labels_train)

combined_features_labels_test = list(zip(combined_features_test, combined_labels_test))
random.shuffle(combined_features_labels_test)
combined_features_test, combined_labels_test = zip(*combined_features_labels_test)
combined_features_test = np.array(combined_features_test)
combined_labels_test = np.array(combined_labels_test)

print("Shape features: ", np.shape(tau_features))
print("Shape: labels ", np.shape(tau_labels))

def create_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(640, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(640, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(320, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(160, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(128, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(32, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dropout(0.05))
    model.add(tf.keras.layers.Dense(8, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(4))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=0.0001),
        loss=tf.keras.losses.mean_squared_error
    )
    return model


model = create_model()

model.fit(
    combined_features_train,
    combined_labels_train,
    batch_size=20,
    epochs=150,
    validation_data=(combined_features_test, combined_labels_test)
)
model.evaluate(
    combined_features_test,
    combined_labels_test,
    batch_size=10
)


pred = model.predict(
    combined_features_test
)

np.savetxt('model_unify.csv', np.array(pred), delimiter=',')
np.savetxt('unify_pions.csv', combined_features_test, delimiter=',')

print(pred)
