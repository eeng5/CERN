import tensorflow as tf
import numpy as np
from array import array
import math

tau_features = np.loadtxt('tau_data/15GeV/15gev_tau_features.csv', delimiter=',')
tau_labels = np.loadtxt('tau_data/15GeV/15gev_tau_labels.csv', delimiter=',')
antitau_features = np.loadtxt('tau_data/15GeV/15gev_antitau_features.csv', delimiter=',')
antitau_labels = np.loadtxt('tau_data/15GeV/15gev_antitau_labels.csv', delimiter=',')

tau_features = np.concatenate((tau_features, np.array(np.loadtxt('tau_data/9.5GeV/9_5gev_tau_features.csv', delimiter=','))), axis=0)
tau_labels = np.concatenate((tau_labels, np.array(np.loadtxt('tau_data/9.5GeV/9_5gev_tau_labels.csv', delimiter=','))), axis = 0)
antitau_features = np.concatenate((antitau_features, np.array(np.loadtxt('tau_data/9.5GeV/9_5gev_tau_features.csv', delimiter=','))), axis=0)
antitau_labels = np.concatenate((antitau_labels, np.array(np.loadtxt('tau_data/9.5GeV/9_5gev_antitau_labels.csv', delimiter=','))), axis=0)

tau_features = np.concatenate((tau_features, np.array(np.loadtxt('tau_data/5GeV/5gev_tau_features.csv', delimiter=','))), axis=0)
tau_labels = np.concatenate((tau_labels,np.array( np.loadtxt('tau_data/5GeV/5gev_tau_labels.csv', delimiter=','))), axis=0)
antitau_features = np.concatenate((antitau_features, np.array(np.loadtxt('tau_data/5GeV/5gev_tau_features.csv', delimiter=','))), axis = 0)
antitau_labels = np.concatenate((antitau_labels, np.array(np.loadtxt('tau_data/5GeV/5gev_antitau_labels.csv', delimiter=','))), axis = 0)

# indices = []
# anti_indices = []
# for i in range(len(tau_features)):
#     indices.append(i)
# indices = tf.random.shuffle(indices)
# tau_features = tf.gather(tau_features, indices)
# tau_labels = tf.gather(tau_labels, indices)
# for i in range(len(antitau_features)):
#     anti_indices.append(i)
# antitau_features = tf.gather(antitau_features, anti_indices)
# antitau_labels = tf.gather(antitau_labels, anti_indices)

indices = []
shuffled_tau_data = [0 for i in range(len(tau_features))]
shuffled_tau_labels = [0 for i in range(len(tau_labels))]
shuffled_antitau_data = [0 for i in range(len(antitau_features))]
shuffled_antitau_labels = [0 for i in range(len(antitau_labels))]
for i in range(len(tau_features)):
    indices.append(i)
np.random.shuffle(indices)
for index, (_, x) in enumerate(np.ndenumerate(indices)):
    shuffled_tau_data[x] = tau_features[index]
    shuffled_tau_labels[x] = tau_labels[index]
    shuffled_antitau_data[x] = antitau_features[index]
    shuffled_antitau_labels[x] = antitau_labels[index]
split_index = math.floor(0.9 * len(tau_features))
tau_features = np.array(shuffled_tau_data)
antitau_features = np.array(shuffled_antitau_data)
tau_labels = np.array(shuffled_tau_labels)
antitau_labels = np.array(shuffled_antitau_labels)

tau_features_train = tau_features[0: int(0.9 * tau_features.shape[0]), :]
tau_features_test = tau_features[int(0.9 * tau_features.shape[0]):, :]
tau_labels_train = tau_labels[0: int(0.9 * tau_labels.shape[0]), :]
tau_labels_test = tau_labels[int(0.9 * tau_labels.shape[0]):, :]

antitau_features_train = antitau_features[0: int(0.9 * antitau_features.shape[0]), :]
antitau_features_test = antitau_features[int(0.9 * antitau_features.shape[0]):, :]
antitau_labels_train = antitau_labels[0: int(0.9 * antitau_labels.shape[0]), :]
antitau_labels_test = antitau_labels[int(0.9 * antitau_labels.shape[0]):, :]

#Normalize transfer momentum (p_t) but for only features
#(num_features, 3) -> (num_features, 1)
#tau_features_train[:,0] / tau_train_total_pt

# tau_train_total_pt = np.sum(tau_features_train[:, [0, 3, 6]], axis=1)
# tau_test_total_pt = np.sum(tau_features_test[:, [0, 3, 6]], axis=1)
# antitau_train_total_pt = np.sum(antitau_features_train[:, [0, 3, 6]], axis=1)
# antitau_test_total_pt = np.sum(antitau_features_test[:, [0, 3, 6]], axis=1)

tau_train_total_pt = np.sum(np.sum((tau_features_train[:,0], tau_features_train[:,3], tau_features_train[:,6]), axis=1), axis = 0)
tau_test_total_pt = np.sum(np.sum((tau_features_test[:,0], tau_features_test[:,3], tau_features_test[:,6]), axis=1), axis=0)
antitau_train_total_pt = np.sum(np.sum((antitau_features_train[:,0], antitau_features_train[:,3], antitau_features_train[:,6]), axis=1), axis = 0)
antitau_test_total_pt = np.sum(np.sum((antitau_features_test[:,0], antitau_features_test[:,3], antitau_features_test[:,6]), axis=1), axis = 0)

# tau_train_total_pt = np.transpose(tau_train_total_pt)
# tau_test_total_pt = np.transpose(tau_test_total_pt)
# antitau_train_total_pt = np.transpose(antitau_train_total_pt)
# antitau_test_total_pt = np.transpose(antitau_test_total_pt)

tf.compat.v1.enable_eager_execution()
print("scalar shape", tau_train_total_pt.shape)
tau_features_train[:,0] = tau_features_train[:,0] / tau_train_total_pt #error on this line now
tau_features_train[:,3] = tau_features_train[:,3] / tau_train_total_pt
tau_features_train[:,6] = tau_features_train[:,6] / tau_train_total_pt

tau_features_test[:,0] = tau_features_test[:,0] / tau_test_total_pt
tau_features_test[:,3] = tau_features_test[:,3] / tau_test_total_pt
tau_features_test[:,6] = tau_features_test[:,6] / tau_test_total_pt

antitau_features_train[:,0] = antitau_features_train[:,0] / antitau_train_total_pt
antitau_features_train[:,3] = antitau_features_train[:,3] / antitau_train_total_pt
antitau_features_train[:,6] = antitau_features_train[:,6] / antitau_train_total_pt

antitau_features_test[:,0] = antitau_features_test[:,0] / antitau_test_total_pt
antitau_features_test[:,3] = antitau_features_test[:,3] / antitau_test_total_pt
antitau_features_test[:,6] = antitau_features_test[:,6] / antitau_test_total_pt

#Original model loss with 150 epochs and with 1000 layers:
#Original model loss with 150 epochs and without 1000 layers: 0.0114, 0.0366
#Best Tanh loss without 1000 layers:
#L1 and L2 regularization loss without 1000 layers:
#Normalized P_t loss:
#Normalized angle loss:
#Unifying tau and anittau loss:
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
np.savetxt('tau_pions_pt_normalization.csv', tau_features_test, delimiter=',')
np.savetxt('antitau_pions_pt_normalization.csv', antitau_features_test, delimiter=',')
np.savetxt('tau_model_pt_normalization.csv', np.array(pred), delimiter=',')
np.savetxt('antitau_model_pt_normalization.csv', np.array(anti_pred), delimiter=',')

print(pred)
print(anti_pred)
