import tensorflow as tf
import numpy as np
from array import array
from keras import losses
import keras as keras
import keras.backend as K
from functools import partial

#concatenated_stuff = np.concatenate((tau_labels_train, tau_labels_train, tau_labels_train), axis=1)
# print(tf.keras.losses.mean_squared_error(concatenated_stuff, tau_features_train).shape)
# print(loss_function(tau_features_train, tau_labels_train))
# print(loss_function(antitau_features_train, antitau_labels_train))

#Original model loss with 150 epochs and with 1000 layers: 0.0261, 0.0602
#Original model loss with 150 epochs and without 1000 layers: 0.0114, 0.0366
#Best Tanh loss without 1000 layers:
#L1 and L2 regularization loss without 1000 layers:
#Normalized P_t loss for 10 epochs: 1.2924, 1.2562
#Normalized angle loss:
#Unifying tau and anittau loss: 0.5207

class Model(tf.keras.Model):
    def __init__(self):
        """
        The model for the generator network is defined here.
        """
        super(Model, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0.5)
        self.layer1 = tf.keras.layers.Dense(640, activation="relu")
        self.layer2 = tf.keras.layers.Dense(640, activation="relu")
        self.layer3 = tf.keras.layers.Dense(320, activation="relu")
        self.layer4 = tf.keras.layers.Dense(160, activation="relu")
        self.layer5 = tf.keras.layers.Dense(128, activation="relu")
        self.layer6 = tf.keras.layers.Dense(64, activation="relu")
        self.layer7 = tf.keras.layers.Dense(32, activation="relu")
        self.layer8 = tf.keras.layers.Dense(8, activation="relu")
        self.layer9 = tf.keras.layers.Dense(3, activation="relu")

        self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.dropout3 = tf.keras.layers.Dropout(0.1)
        self.dropout4 = tf.keras.layers.Dropout(0.1)
        self.dropout5 = tf.keras.layers.Dropout(0.1)
        self.dropout6 = tf.keras.layers.Dropout(0.1)
        self.dropout7 = tf.keras.layers.Dropout(0.5)

        self.batch_size = 100

    @tf.function
    def call(self, inputs):

        output_layer1 = self.layer1(inputs)
        output_layer1 = self.dropout1(output_layer1)

        output_layer2 = self.layer2(output_layer1)
        output_layer2 = self.dropout2(output_layer2)

        output_layer3 = self.layer3(output_layer2)
        output_layer3 = self.dropout3(output_layer3)

        output_layer4 = self.layer4(output_layer3)
        output_layer4 = self.dropout4(output_layer4)

        output_layer5 = self.layer5(output_layer4)
        output_layer5 = self.dropout5(output_layer5)

        output_layer6 = self.layer6(output_layer5)
        output_layer6 = self.dropout6(output_layer6)

        output_layer7 = self.layer7(output_layer6)
        output_layer7 = self.dropout7(output_layer7)

        output_layer8 = self.layer8(output_layer7)
        output_layer9 = self.layer9(output_layer8)

        return output_layer9

    @tf.function
    def loss_function(self, data, labels):
        # print("tau: ", data.shape)

        MSE_pt = tf.math.divide(tf.math.reduce_sum((data[:,0] - labels[:,0])**2), self.batch_size)
        MSE_eta = tf.math.divide(tf.math.reduce_sum((data[:,1] - labels[:,1])**2), self.batch_size)
        MSE_phi = tf.math.divide(tf.math.reduce_sum((data[:,2] - labels[:,2])**2), self.batch_size)

        loss = 0.5*MSE_pt + 0.25*MSE_eta + 0.25*MSE_phi
        return loss

def train(model, train_inputs, train_labels):
    train_inputs = tf.dtypes.cast(train_inputs, dtype=tf.float32)
    train_labels = tf.dtypes.cast(train_labels, dtype=tf.float32)
    num_batches = len(train_inputs) // model.batch_size
    for i in range(0, len(train_inputs), model.batch_size):
        if len(train_inputs[i:,:]) < model.batch_size:
            break
        inputs = train_inputs[i : i + model.batch_size, :]
        labels = train_labels[i : i + model.batch_size, :]
        with tf.GradientTape() as tape:
            logits = model.call(inputs)
            loss = model.loss_function(logits, labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return logits

def test(model, test_inputs, test_labels):
    test_inputs = tf.dtypes.cast(test_inputs, dtype=tf.float32)
    test_labels = tf.dtypes.cast(test_labels, dtype=tf.float32)
    num_batches = len(test_inputs) // model.batch_size
    total_loss = 0
    counter = 0

    entire_data = []

    for i in range(0, len(test_inputs), model.batch_size):

        if len(test_inputs[i:,:]) < model.batch_size:
            break
        inputs = test_inputs[i : i + model.batch_size, :]
        labels = test_labels[i : i + model.batch_size, :]
        with tf.GradientTape() as tape:
            logits = model.call(inputs)
            loss = model.loss_function(logits, labels)
            total_loss += loss
            counter += 1
        batch_data = tf.concat((inputs, logits), axis=1)
        entire_data.append(batch_data)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    res_tensor = entire_data[0]
    for i in range(1, len(entire_data)):
        res_tensor = tf.concat((res_tensor, entire_data[i]), axis=0)
    print(res_tensor.shape)
    # print("Total Loss", total_loss / counter)
    return (res_tensor, total_loss / counter)

def main():
    tau_features = np.loadtxt('15gev_tau_features.csv', delimiter=',')
    tau_labels = np.loadtxt('15gev_tau_labels.csv', delimiter=',')
    antitau_features = np.loadtxt('15gev_antitau_features.csv', delimiter=',')
    antitau_labels = np.loadtxt('15gev_antitau_labels.csv', delimiter=',')

    tau_features = np.concatenate((tau_features, np.array(np.loadtxt('9_5gev_tau_features.csv', delimiter=','))), axis=0)
    tau_labels = np.concatenate((tau_labels, np.array(np.loadtxt('9_5gev_tau_labels.csv', delimiter=','))), axis = 0)
    antitau_features = np.concatenate((antitau_features, np.array(np.loadtxt('9_5gev_tau_features.csv', delimiter=','))), axis=0)
    antitau_labels = np.concatenate((antitau_labels, np.array(np.loadtxt('9_5gev_antitau_labels.csv', delimiter=','))), axis=0)

    tau_features = np.concatenate((tau_features, np.array(np.loadtxt('5gev_tau_features.csv', delimiter=','))), axis=0)
    tau_labels = np.concatenate((tau_labels,np.array( np.loadtxt('5gev_tau_labels.csv', delimiter=','))), axis=0)
    antitau_features = np.concatenate((antitau_features, np.array(np.loadtxt('5gev_tau_features.csv', delimiter=','))), axis = 0)
    antitau_labels = np.concatenate((antitau_labels, np.array(np.loadtxt('5gev_antitau_labels.csv', delimiter=','))), axis = 0)

    tau_features_train = tau_features[0: int(0.9 * tau_features.shape[0]), :]
    tau_features_test = tau_features[int(0.9 * tau_features.shape[0]):, :]
    tau_labels_train = tau_labels[0: int(0.9 * tau_labels.shape[0]), :]
    tau_labels_test = tau_labels[int(0.9 * tau_labels.shape[0]):, :]

    antitau_features_train = antitau_features[0: int(0.9 * antitau_features.shape[0]), :]
    antitau_features_test = antitau_features[int(0.9 * antitau_features.shape[0]):, :]
    antitau_labels_train = antitau_labels[0: int(0.9 * antitau_labels.shape[0]), :]
    antitau_labels_test = antitau_labels[int(0.9 * antitau_labels.shape[0]):, :]

    indices = []
    shuffled_tau_data = [0 for i in range(len(tau_features_train))]
    shuffled_tau_labels = [0 for i in range(len(tau_labels_train))]
    shuffled_antitau_data = [0 for i in range(len(antitau_features_train))]
    shuffled_antitau_labels = [0 for i in range(len(antitau_labels_train))]
    for i in range(len(tau_features_train)):
        indices.append(i)
    np.random.shuffle(indices)
    for index, (_, x) in enumerate(np.ndenumerate(indices)):
        shuffled_tau_data[x] = tau_features_train[index]
        shuffled_tau_labels[x] = tau_labels_train[index]
        shuffled_antitau_data[x] = antitau_features_train[index]
        shuffled_antitau_labels[x] = antitau_labels_train[index]
    split_index = math.floor(0.9 * len(tau_features_train))
    tau_features_train = np.array(shuffled_tau_data)
    antitau_features_train = np.array(shuffled_antitau_data)
    tau_labels_train = np.array(shuffled_tau_labels)
    antitau_labels_train = np.array(shuffled_antitau_labels)

    tau_model = Model()
    antitau_model = Model()
    for i in range(150):
        print("EPOCH: " + str(i+1))
        tau_logits =  train(tau_model, tau_features_train, tau_labels_train)
        tau_res, tau_loss = test(tau_model, tau_features_test, tau_labels_test)
        anti_logits = train(antitau_model, antitau_features_train, antitau_labels_train)
        antitau_res, antitau_loss = test(antitau_model, antitau_features_test, antitau_labels_test)
        print("tau_loss: " + str(tau_loss))
        print("antitau_loss: " + str(antitau_loss))
        if i == 149:
            np.savetxt('tau_orig_model.csv', np.array(tau_res), delimiter=',')
            np.savetxt('antitau_orig_model.csv',np.array(antitau_res), delimiter=',')
    # tau_features = tf.stack((tau_features, tau_res), axis=1)
   # tau_features = np.append(tau_features, tau_logits, axis=0)
    # antitau_features = tf.stack((antitau_features, anti_logits), axis=1)
   # antitau_features = np.append(antitau_features, anti_logits, axis=0)
   # tau_features = tf.stack((tau_features, tau_logits), axis=1)
   # antitau_features = tf.stack((antitau_features, anti_logits), axis=1)

    # np.savetxt('tau_orig_model.csv', tau_features )
    # np.savetxt('antitau_orig_model.csv', antitau_features )
if __name__ == '__main__':
   main()
