import tensorflow as tf
import numpy as np
from array import array
import math

#Normalize angles but only features



#Original model loss with 150 epochs and with 1000 layers: 0.0261, 0.0602
#Original model loss with 150 epochs and without 1000 layers: 0.0114, 0.0366
#Best Tanh loss without 1000 layers:
#L1 and L2 regularization loss without 1000 layers:
#Normalized P_t loss for 10 epochs: 1.2924, 1.2562
#Normalized angle loss:
#Unifying tau and anittau loss: 0.5207
def outermost_angle_identifier(phi_1, phi_2, phi_3):
    phi_1_minus_phi_2 = phi_1 - phi_2
    if phi_1_minus_phi_2 < -math.pi:
        phi_1_minus_phi_2 += math.pi*2
    phi_2_minus_phi_3 = phi_2 - phi_3
    if phi_2_minus_phi_3 < -math.pi:
        phi_2_minus_phi_3 += math.pi*2
    phi_3_minus_phi_1 = phi_3 - phi_1
    if phi_3_minus_phi_1 < -math.pi:
        phi_3_minus_phi_1 += math.pi*2
    max_angle_difference = max(abs(phi_1_minus_phi_2), abs(phi_2_minus_phi_3), abs(phi_3_minus_phi_1))

    if max_angle_difference == abs(phi_1_minus_phi_2):
        return (phi_1, phi_2, phi_3)
    if max_angle_difference == abs(phi_2_minus_phi_3):
        return (phi_2, phi_3, phi_1)
    if max_angle_difference == abs(phi_3_minus_phi_1):
        return (phi_3, phi_1, phi_2)
#test cases
# phi_1_test = math.radians(-150)
# phi_2_test = math.radians(-175)
# phi_3_test = math.radians(120)
# print(outermost_angle_identifier(phi_1_test, phi_2_test, phi_3_test)) #(-150, 120)


def angle_normalization(phi_1, phi_2, phi_3):
    phi_extreme_1, phi_extreme_2, phi_inner = outermost_angle_identifier(phi_1, phi_2, phi_3)

    # phi extreme normalization
    phi_extreme_1_adj = math.pi - abs(phi_extreme_1)
    phi_extreme_2_adj = math.pi - abs(phi_extreme_2)

    # handle zero case
    if phi_extreme_1 == 0:
        phi_extreme_1_adj = 0
        phi_extreme_2_adj = phi_extreme_2
    if phi_extreme_2 == 0:
        phi_extreme_2_adj = 0
        phi_extreme_1_adj = phi_extreme_1

    difference = phi_extreme_1_adj + phi_extreme_2_adj

    # unadjusted
    phi_ex_og_1 = phi_extreme_1
    phi_ex_og_2 = phi_extreme_2
    phi_inner_og = phi_inner

    if phi_extreme_1 < 0:
        phi_extreme_1 += math.pi*2
    if phi_extreme_2 < 0:
        phi_extreme_2 += math.pi*2
    maximum = max(phi_extreme_1, phi_extreme_2)
    if maximum == phi_extreme_1:
        phi_extreme_1_adj = difference / 2
        phi_extreme_2_adj = - difference / 2
    elif maximum == phi_extreme_2:
        phi_extreme_1_adj = - difference / 2
        phi_extreme_2_adj = difference / 2

    # phi inner normalization
    mean_angle = (phi_extreme_1 + phi_extreme_2) / 2
    if phi_inner < 0:
        phi_inner += math.pi*2
    if phi_inner < mean_angle:
        phi_inner = mean_angle - phi_inner
    elif phi_inner > mean_angle:
        phi_inner = phi_inner - mean_angle

    # assign old variables
    if phi_ex_og_1 == phi_1 and phi_ex_og_2 == phi_2:
        phi_1 = phi_extreme_1_adj
        phi_2 = phi_extreme_2_adj
        phi_3 = phi_inner
    elif phi_ex_og_1 == phi_1 and phi_ex_og_2 == phi_3:
        phi_1 = phi_extreme_1_adj
        phi_2 = phi_inner
        phi_3 = phi_extreme_2_adj
    elif phi_ex_og_1 == phi_2 and phi_ex_og_2 == phi_3:
        phi_1 = phi_inner
        phi_2 = phi_extreme_1_adj
        phi_3 = phi_extreme_2_adj
    elif phi_ex_og_1 == phi_2 and phi_ex_og_2 == phi_1:
        phi_1 = phi_extreme_2_adj
        phi_2 = phi_extreme_1_adj
        phi_3 = phi_inner
    elif phi_ex_og_1 == phi_3 and phi_ex_og_2 == phi_1:
        phi_1 = phi_extreme_2_adj
        phi_2 = phi_inner
        phi_3 = phi_extreme_1_adj
    elif phi_ex_og_1 == phi_3 and phi_ex_og_2 == phi_2:
        phi_1 = phi_inner
        phi_2 = phi_extreme_2_adj
        phi_3 = phi_extreme_1_adj

    return (phi_1, phi_2, phi_3)

#phi_test_test_1 = 0
#phi_test_test_2 = math.radians(20)
#phi_test_test_3 = math.radians(60)
# 0.5236 for a pair , -0.1745
#print("TEST: " + str(angle_normalization(phi_test_test_1, phi_test_test_2, phi_test_test_3)))

def normalize_phi(phi_data):
    shape = phi_data.shape.as_list()
    #print("shape: ", shape)
    res = np.empty((shape[0], shape[1]))
    # phi_data is (batch, 3)
    for row in range(shape[0]):
        just_row = phi_data[row,:].numpy()
        #print(just_row)
        phi_1, phi_2, phi_3 = just_row[0], just_row[1], just_row[2]
        normalized_phi_1, normalized_phi_2, normalized_phi_3 = angle_normalization(phi_1, phi_2, phi_3)
        np_row = np.asarray([[normalized_phi_1, normalized_phi_2, normalized_phi_3]])
        # print("np_row: ", np_row.shape)
        # print("res: ", res.shape)
        res[row:,] = np_row
    return tf.convert_to_tensor(res, dtype=tf.float32)

# extracted_phi_data = tf.stack((tau_features_train[:,2], tau_features_train[:,5], tau_features_train[:,8]), axis=1)

#test cases
# phi_1_test = math.radians(-150) # -150
# phi_2_test = math.radians(-175) # -175
# phi_3_test = math.radians(120) # 120
#
# print(angle_normalization(phi_1_test, phi_2_test, phi_3_test))

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

    def call(self, inputs):
        # normalize phi values
        extracted_phis = tf.stack((inputs[:,2], inputs[:,5], inputs[:,8]), axis=1)
        normalized_phis = normalize_phi(extracted_phis)

        new_inputs = tf.stack((inputs[:,0], inputs[:,1], normalized_phis[:,0], inputs[:,3], inputs[:,4], normalized_phis[:,1], inputs[:,6], inputs[:,7], normalized_phis[:,2]), axis=1)

        output_layer1 = self.layer1(new_inputs)
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

    def loss_function(self, data, labels):
        loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(labels, data))
        return loss

def train(model, train_inputs, train_labels):
    train_inputs = tf.dtypes.cast(train_inputs, dtype=tf.float32)
    train_labels = tf.dtypes.cast(train_labels, dtype=tf.float32)
    num_batches = len(train_inputs) // model.batch_size
    for i in range(0, len(train_inputs), model.batch_size):
        inputs = train_inputs[i : i + model.batch_size, :]
        labels = train_labels[i : i + model.batch_size, :]
        if len(inputs) < model.batch_size:
            break
        with tf.GradientTape() as tape:
            logits = model.call(inputs)
            loss = model.loss_function(logits, labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_inputs, test_labels): #new_test with
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
    # tf.compat.v1.enable_eager_execution()
    tau_features = np.loadtxt('15gev_tau_features.csv', delimiter=',')
    tau_labels = np.loadtxt('15gev_tau_labels.csv', delimiter=',')
    antitau_features = np.loadtxt('15gev_antitau_features.csv', delimiter=',')
    antitau_labels = np.loadtxt('15gev_antitau_labels.csv', delimiter=',')

    tau_features = np.concatenate((tau_features, np.array(np.loadtxt('9_5gev_tau_features.csv', delimiter=','))), axis=0)
    tau_labels = np.concatenate((tau_labels, np.array(np.loadtxt('9_5gev_tau_labels.csv', delimiter=','))), axis = 0)
    antitau_features = np.concatenate((antitau_features, np.array(np.loadtxt('9_5gev_antitau_features.csv', delimiter=','))), axis=0)
    antitau_labels = np.concatenate((antitau_labels, np.array(np.loadtxt('9_5gev_antitau_labels.csv', delimiter=','))), axis=0)

    tau_features = np.concatenate((tau_features, np.array(np.loadtxt('5gev_tau_features.csv', delimiter=','))), axis=0)
    tau_labels = np.concatenate((tau_labels,np.array( np.loadtxt('5gev_tau_labels.csv', delimiter=','))), axis=0)
    antitau_features = np.concatenate((antitau_features, np.array(np.loadtxt('5gev_antitau_features.csv', delimiter=','))), axis = 0)
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
            np.savetxt('tau_model_angle_normalization.csv', np.array(tau_res), delimiter=',')
            np.savetxt('antitau_model_angle_normalization.csv',np.array(antitau_res), delimiter=',')


if __name__ == '__main__':
   main()
