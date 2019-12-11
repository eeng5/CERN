import tensorflow as tf
import numpy as np
from array import array

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


def create_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(640, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(1280, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2560, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1280, activation=tf.keras.activations.relu))
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
    model.add(tf.keras.layers.Dense(3))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=0.0001),
        loss=tf.keras.losses.mean_squared_error
    )
    return model


tau_model = create_model()
antitau_model = create_model()

tau_model.fit(
    tau_features_train,
    tau_labels_train,
    batch_size=20,
    epochs=150,
    validation_data=(tau_features_test, tau_labels_test)
    # validation_data=None
)
tau_model.evaluate(
    tau_features_test,
    tau_labels_test,
    batch_size=10
)
antitau_model.fit(
    antitau_features_train,
    antitau_labels_train,
    batch_size=20,
    epochs=150,
    validation_data=(antitau_features_test, antitau_labels_test)
    # validation_data = None
)
antitau_model.evaluate(
    antitau_features_test,
    antitau_labels_test,
    batch_size=10
)

pred = tau_model.predict(
    tau_features_test
)
anti_pred = antitau_model.predict(
    antitau_features_test
)

np.savetxt('tau_pions_orig.csv', tau_features_test, delimiter=',')
np.savetxt('antitau_pions_orig.csv', antitau_features_test, delimiter=',')
np.savetxt('tau_model_orig.csv', np.array(pred), delimiter=',')
np.savetxt('antitau_model_orig.csv',np.array(anti_pred), delimiter=',')

print(pred)
print(anti_pred)
