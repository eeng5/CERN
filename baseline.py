import xgboost as xgb
from xgboost import XGBClassifier
import numpy as np
import math
import sklearn as sklearn
#from sklearn.metrics import accuracy_score

# class sklearn.multioutput.MultiOutputRegressor(estimator, n_jobs=None)

# def get_data(tau_data_file_name, tau_labels_file_name, antitau_data_file_name, antitau_labels_file_name):
#     tau_data = np.loadtxt(tau_data_file_name, delimiter=',') #returns numpy array with the data
#     antitau_data = np.loadtxt(antitau_data_file_name, delimiter=',')
#     tau_labels = np.loadtxt(tau_labels_file_name, delimiter=',') #returns numpy of labels
#     antitau_labels = np.loadtxt(antitau_labels_file_name, delimiter=',')
#
#     data = np.concatenate((tau_data, antitau_data), axis=0)
#     labels = np.concatenate((tau_labels, antitau_labels), axis=0)
#     return data,labels

def main():
    tau_labels_15 = 'tau_data/15GeV/15gev_tau_labels.csv'
    tau_features_15 = 'tau_data/15GeV/15gev_tau_features.csv'
    antitau_labels_15 = 'tau_data/15GeV/15gev_antitau_labels.csv'
    antitau_features_15 = 'tau_data/15GeV/15gev_antitau_features.csv'

    tau_labels_9_5 = 'tau_data/9.5GeV/9_5gev_tau_labels.csv'
    tau_features_9_5 = 'tau_data/9.5GeV/9_5gev_tau_features.csv'
    antitau_labels_9_5 = 'tau_data/9.5GeV/9_5gev_antitau_labels.csv'
    antitau_features_9_5 = 'tau_data/9.5GeV/9_5gev_antitau_features.csv'

    tau_labels_5 = 'tau_data/5GeV/5gev_tau_labels.csv'
    tau_features_5 = 'tau_data/5GeV/5gev_tau_features.csv'
    antitau_labels_5 = 'tau_data/5GeV/5gev_antitau_labels.csv'
    antitau_features_5 = 'tau_data/5GeV/5gev_antitau_features.csv'

    #tentative
    tau_data = np.loadtxt(tau_features_15, delimiter=',') #returns numpy array with the data
    antitau_data = np.loadtxt(antitau_features_15, delimiter=',')
    tau_labels = np.loadtxt(tau_labels_15, delimiter=',') #returns numpy of labels
    antitau_labels = np.loadtxt(antitau_labels_15, delimiter=',')

    tau_data = np.concatenate((tau_data, np.loadtxt(tau_features_9_5, delimiter=',')), axis=0)#returns numpy array with the data
    antitau_data = np.concatenate((antitau_data, np.loadtxt(antitau_features_9_5, delimiter=',')), axis=0)
    tau_labels = np.concatenate((tau_labels, np.loadtxt(tau_labels_9_5, delimiter=',')), axis=0) #returns numpy of labels
    antitau_labels = np.concatenate((antitau_labels, np.loadtxt(antitau_labels_9_5, delimiter=',')), axis=0)

    tau_data = np.concatenate((tau_data, np.loadtxt(tau_features_5, delimiter=',')), axis=0)#returns numpy array with the data
    antitau_data = np.concatenate((antitau_data, np.loadtxt(antitau_features_5, delimiter=',')), axis=0)
    tau_labels = np.concatenate((tau_labels, np.loadtxt(tau_labels_5, delimiter=',')), axis=0) #returns numpy of labels
    antitau_labels = np.concatenate((antitau_labels, np.loadtxt(antitau_labels_5, delimiter=',')), axis=0)

    indices = []
    shuffled_tau_data = [0 for i in range(len(tau_data))]
    shuffled_tau_labels = [0 for i in range(len(tau_labels))]
    shuffled_antitau_data = [0 for i in range(len(antitau_data))]
    shuffled_antitau_labels = [0 for i in range(len(antitau_labels))]
    for i in range(len(tau_data)):
        indices.append(i)
    np.random.shuffle(indices)
    for index, (_, x) in enumerate(np.ndenumerate(indices)):
        shuffled_tau_data[x] = tau_data[index]
        shuffled_tau_labels[x] = tau_labels[index]

        shuffled_antitau_data[x] = antitau_data[index]
        shuffled_antitau_labels[x] = antitau_labels[index]
    split_index = math.floor(0.9 * len(tau_data))

    shuffled_tau_data = np.array(shuffled_tau_data)
    shuffled_tau_labels = np.array(shuffled_tau_labels)
    shuffled_antitau_data = np.array(shuffled_antitau_data)
    shuffled_antitau_labels = np.array(shuffled_antitau_labels)
    # divide into training and testing sets
    tau_train_data = shuffled_tau_data[:split_index,]
    tau_test_data = shuffled_tau_data[split_index:,]
    tau_train_labels = shuffled_tau_labels[:split_index,]
    tau_test_labels = shuffled_tau_labels[split_index:,]

    antitau_train_data = shuffled_antitau_data[:split_index,]
    antitau_test_data = shuffled_antitau_data[split_index:,]
    antitau_train_labels = shuffled_antitau_labels[:split_index,]
    antitau_test_labels = shuffled_antitau_labels[split_index:,]

    np.savetxt("tau_data_features.csv", tau_test_data, delimiter=',')
    np.savetxt("antitau_data_features.csv", antitau_test_data, delimiter=',')

    param = { 'max_depth':2, 'eta':1}

    mean_squared_error_sum = 0
    for i in range(3):
        column_labels_train = tau_train_labels[:,i]
        column_labels_test = tau_test_labels[:,i]
        dtrain = xgb.DMatrix(tau_train_data, column_labels_train)
        dtest = xgb.DMatrix(tau_test_data, column_labels_test)
        model = xgb.train(param, dtrain)
        crap = model.predict(dtrain)
        np.savetxt("tau_data" + str(i) + ".csv", crap, delimiter=',')
        #mean_squared_error_sum +=
        print(model.eval(dtest))

    for i in range(3):
        column_labels_train = antitau_train_labels[:,i]
        column_labels_test = antitau_test_labels[:,i]
        dtrain = xgb.DMatrix(antitau_train_data, column_labels_train)
        dtest = xgb.DMatrix(antitau_test_data, column_labels_test)
        model = xgb.train(param, dtrain)
        crap = model.predict(dtrain)
        np.savetxt("antitau_data" + str(i) + ".csv", crap, delimiter=',')
        #mean_squared_error_sum +=
        print(model.eval(dtest))
    #print("Mean Squared Error:" + str(mean_squared_error_sum / 3))

    # model = XGBClassifier()
    # model.fit(train_data)
    # predictions = model.predict(test_data)
    # score = [round(value) for value in predictions]
    # accuracy = sklearn.accuracy_score(predictions, score)
    # print("Accuracy: " + str(accuracy))

    # use xgb boost stuff
    #weights = np.random.rand(len(data), 3)
    #param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic'}
    #dtrain = xgb.DMatrix(train_data, label=train_labels, missing=-999.0, weight=weights)
    #dtest = xgb.DMatrix(test_data, label=test_labels, missing=-999.0, weight=weights)
    #num_round = 10
    #evallist = [(dtest, 'eval'), (dtrain, 'train')]
    #bst = xgb.train(param, dtrain, num_round, evallist)

if __name__ == '__main__':
 	main()


#EXAMPLE
# # load data
# tau_data = np.loadtxt(tau_data_file_name, delimiter=',') #returns numpy array with the data
# antitau_data = np.loadtxt(antitau_data_file_name, delimiter=',')
# tau_labels = np.loadtxt(tau_labels_file_name, delimiter=',') #returns numpy of labels
# antitau_labels = np.loadtxt(antitau_labels_file_name, delimiter=',')
# data = np.concatenate((tau_data, antitau_data), axis=0)
# labels = np.concatenate((tau_labels, antitau_labels), axis=0)
# # split data into train and test sets
# seed = 7
# test_size = 0.33
# data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=test_size, random_state=seed)
# # fit model no training data
# model = XGBClassifier()
# model.fit(data_train, labels_train)
# # make predictions for test data
# pred = model.predict(data_test)
# predictions = [round(value) for value in pred]
# # evaluate predictions
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))


#FROM XGBOOST DOCS
# read in data
# dtrain = xgb.DMatrix('demo/data/agaricus.txt.train')
# dtest = xgb.DMatrix('demo/data/agaricus.txt.test')
# specify parameters via map
# param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic' }
# num_round = 2
# bst = xgb.train(param, dtrain, num_round)
# make prediction
#preds = bst.predict(dtest)
