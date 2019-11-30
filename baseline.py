import xgboost as xgb
from xgboost import XGBClassifier
import numpy as np
import math
#from sklearn.metrics import accuracy_score

# class sklearn.multioutput.MultiOutputRegressor(estimator, n_jobs=None)

def get_data(tau_data_file_name, tau_labels_file_name, antitau_data_file_name, antitau_labels_file_name):
    tau_data = np.loadtxt(tau_data_file_name, delimiter=',') #returns numpy array with the data
    antitau_data = np.loadtxt(antitau_data_file_name, delimiter=',')
    tau_labels = np.loadtxt(tau_labels_file_name, delimiter=',') #returns numpy of labels
    antitau_labels = np.loadtxt(antitau_labels_file_name, delimiter=',')
    #print(tau_data)
    #print(antitau_data)
    data = np.concatenate((tau_data, antitau_data), axis=0)
    labels = np.concatenate((tau_labels, antitau_labels), axis=0)
    return data,labels

def main():
    tau_labels = 'tau_data/15GeV/15gev_tau_labels.csv'
    tau_features = 'tau_data/15GeV/15gev_tau_features.csv'
    antitau_labels = 'tau_data/15GeV/15gev_antitau_labels.csv'
    antitau_features = 'tau_data/15GeV/15gev_antitau_features.csv'
    data, labels = get_data(tau_features, tau_labels, antitau_features, antitau_labels)
    split_index = math.floor(0.9 * len(data))
    print(split_index)
    # divide into training and testing sets
    train_data = data[:split_index,]
    test_data = data[split_index:,]
    train_labels = labels[:split_index,]
    test_labels = labels[split_index:,]

    model = XGBClassifier()
    model.fit(train_data)
    predictions = model.predict(test_data)
    score = [round(value) for value in predictions]
    accuracy = sklearn.accuracy_score(predictions, score)
    print("Accuracy: " + str(accuracy))

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
