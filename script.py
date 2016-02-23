import sys
import random
import scipy.io as spio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns;
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix
import numpy as np
import random as random
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import zero_one_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
# logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model, decomposition
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
# preprocessing library
from sklearn import preprocessing

# util functions
# plot_confusion_matrix from 
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
genres = ['Blues', 'Classical', 'Country', 'Disco', 'Hiphop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
# genres = ['Classical', 'Country', 'Disco', 'Hiphop', 'Jazz', 'Metal', 'Pop', 'Reggae']
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(genres))
    plt.xticks(tick_marks, genres, rotation=45)
    plt.yticks(tick_marks, genres)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# setup
FVs = '/Users/abkhanna/Documents/MATLAB/FV.mat'
LBs = '/Users/abkhanna/Documents/MATLAB/LB.mat'
mfcc = spio.loadmat(FVs)['FV']
labels = spio.loadmat(LBs)['LB'][0]
N = mfcc.shape[1]
maxLearners = 110 
maxDepth = 31

def analysis_ideal_random_forest():
    k = 10
    skf = StratifiedKFold(labels,n_folds=k)
    averageError = 0.0
    avg_cm = np.zeros((len(genres),len(genres)))
    for train_index, test_index in skf:
        X_train, X_test = mfcc[:,train_index], mfcc[:,test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        rf = RandomForestClassifier(n_estimators=maxLearners, max_depth = maxDepth, warm_start = False)
        rf.fit(X_train.T,y_train)
        y_pred = rf.predict(X_test.T)
        error = zero_one_loss(y_pred,y_test)
        avg_cm += confusion_matrix(y_test, y_pred)
        print error
        averageError += (1./k) * error
    print "Average error: %4.2f%s" % (100 * averageError,'%')
    plot_confusion_matrix(avg_cm / k)
    plt.show()

def experiment_depth_random_forest():
    avgError = []
    x_learners = []
    for maxDepth in range(1, 50, 5):
        k = 10
        skf = StratifiedKFold(labels,n_folds=k)
        averageError = 0.0
        for train_index, test_index in skf:
            X_train, X_test = mfcc[:,train_index], mfcc[:,test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            rf = RandomForestClassifier(n_estimators=maxLearners, max_depth = maxDepth, warm_start = False)
            rf.fit(X_train.T,y_train)
            y_pred = rf.predict(X_test.T)
            error = zero_one_loss(y_pred,y_test)
            print error
            averageError += (1./k) * error
        print "Average error: %4.2f%s" % (100 * averageError,'%')
        avgError.append(averageError)
        x_learners.append(maxDepth)
    # graph the errors now.
    plt.plot(x_learners, avgError)
    plt.ylabel('Average Error (k=10)')
    plt.xlabel('Max Depth')
    plt.title('Error as a function of the max depth')
    plt.show()

def experiment_learners_random_forest():
    avgError = []
    x_learners = []
    for maxLearners in range(10, 150, 20):
        k = 10
        skf = StratifiedKFold(labels,n_folds=k)
        averageError = 0.0
        for train_index, test_index in skf:
            X_train, X_test = mfcc[:,train_index], mfcc[:,test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            rf = RandomForestClassifier(n_estimators=maxLearners, max_depth = maxDepth, warm_start = False)
            rf.fit(X_train.T,y_train)
            y_pred = rf.predict(X_test.T)
            error = zero_one_loss(y_pred,y_test)
            print error
            averageError += (1./k) * error
        print "Average error: %4.2f%s" % (100 * averageError,'%')
        avgError.append(averageError)
        x_learners.append(maxLearners)

    plt.plot(x_learners, avgError)
    plt.ylabel('Average Error (k=10)')
    plt.xlabel('Max Learners')
    plt.title('Error as a function of the number of learners')
    plt.show()

def experiment_neighbors_k_nearest_neighbors():
    avgError = []
    x_learners = []
    for k_neighbors in range(1, 20, 1):
        k = 10
        skf = StratifiedKFold(labels,n_folds=k)
        averageError = 0.0
        for train_index, test_index in skf:
            X_train, X_test = mfcc[:,train_index], mfcc[:,test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            knc = KNeighborsClassifier(n_neighbors=k_neighbors, weights='distance')
            knc.fit(X_train.T,y_train)
            y_pred = knc.predict(X_test.T)
            error = zero_one_loss(y_pred,y_test)
            print error
            averageError += (1./k) * error
        print "Average error: %4.2f%s" % (100 * averageError,'%')
        avgError.append(averageError)
        x_learners.append(k_neighbors)

    plt.plot(x_learners, avgError)
    plt.ylabel('Average Error (k=10)')
    plt.xlabel('Number of Neighbors')
    plt.title('Error as a function of the number of neighbors taken into consideration')
    plt.show()

def analysis_neighbors_k_nearest_neighbors():
    k = 10
    skf = StratifiedKFold(labels,n_folds=k)
    averageError = 0.0
    k_neighbors = 3
    avg_cm = np.zeros((len(genres),len(genres)))
    for train_index, test_index in skf:
        X_train, X_test = mfcc[:,train_index], mfcc[:,test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        knc = KNeighborsClassifier(n_neighbors=k_neighbors, weights='distance')
        knc.fit(X_train.T,y_train)
        y_pred = knc.predict(X_test.T)
        error = zero_one_loss(y_pred,y_test)
        avg_cm += confusion_matrix(y_test, y_pred)
        print error
        averageError += (1./k) * error
    print "Average error: %4.2f%s" % (100 * averageError,'%')
    plot_confusion_matrix(avg_cm / k)
    plt.show()

def experiment_pca_logistic_regression():
    logistic = linear_model.LogisticRegression()
    pca = decomposition.PCA()
    pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
    n_components = [20, 40, 64]
    Cs = np.logspace(-4, 4, 3)

    k = 10
    skf = StratifiedKFold(labels,n_folds=k)
    averageError = 0.0
    for train_index, test_index in skf:
        X_train, X_test = mfcc[:,train_index], mfcc[:,test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        estimator = GridSearchCV(pipe,dict(pca__n_components=n_components,
                                  logistic__C=Cs))
        estimator.fit(X_train.T,y_train)
        y_pred = estimator.predict(X_test.T)
        error = zero_one_loss(y_pred,y_test)
        print error
        averageError += (1./k) * error
    print "Average error: %4.2f%s" % (100 * averageError,'%')

def experiment_pca_random_forest():
    avgError = []
    x_learners = []
    pca = decomposition.PCA()
    rf = RandomForestClassifier(n_estimators=maxLearners, max_depth = maxDepth, warm_start = False)
    pipe = Pipeline(steps=[('pca', pca), ('rf', rf)])
    n_components = [60]
    k = 10
    skf = StratifiedKFold(labels,n_folds=k)
    averageError = 0.0
    for train_index, test_index in skf:
        X_train, X_test = mfcc[:,train_index], mfcc[:,test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        estimator = GridSearchCV(pipe, dict(pca__n_components=n_components))
        estimator.fit(X_train.T,y_train)
        y_pred = estimator.predict(X_test.T)
        error = zero_one_loss(y_pred,y_test)
        print error
        averageError += (1./k) * error
    print "Average error: %4.2f%s" % (100 * averageError,'%')

def experiment_pca_n_components_random_forest():
    pca = decomposition.PCA()
    rf = RandomForestClassifier(n_estimators=maxLearners, max_depth = maxDepth, warm_start = False)
    pipe = Pipeline(steps=[('pca', pca), ('rf', rf)])
    avgError = []
    x_learners = []
    for k_components in range(10, 100, 10):
        k = 10
        skf = StratifiedKFold(labels,n_folds=k)
        averageError = 0.0
        for train_index, test_index in skf:
            X_train, X_test = mfcc[:,train_index], mfcc[:,test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            estimator = GridSearchCV(pipe, dict(pca__n_components=[k_components]))
            estimator.fit(X_train.T,y_train)
            y_pred = estimator.predict(X_test.T)
            error = zero_one_loss(y_pred,y_test)
            print error
            averageError += (1./k) * error
        print "Average error: %4.2f%s" % (100 * averageError,'%')
        avgError.append(averageError)
        x_learners.append(k_components)

    plt.plot(x_learners, avgError)
    plt.ylabel('Average Error (k=10)')
    plt.xlabel('Number of Components')
    plt.title('Error as a function of the number of components')
    plt.show()

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

def experiment_estimators_AdaBoostRandomForest():
    avgError = []
    x_learners = []
    rf = RandomForestClassifier(n_estimators=maxLearners, max_depth = maxDepth, warm_start = False)
    for lr in frange(0.01, 1., 0.25):
        k = 10
        skf = StratifiedKFold(labels,n_folds=k)
        averageError = 0.0
        for train_index, test_index in skf:
            X_train, X_test = mfcc[:,train_index], mfcc[:,test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            adb = AdaBoostClassifier(base_estimator=rf, n_estimators=100, learning_rate=lr)
            adb.fit(X_train.T,y_train)
            y_pred = adb.predict(X_test.T)
            error = zero_one_loss(y_pred,y_test)
            print error
            averageError += (1./k) * error
        print "Average error: %4.2f%s" % (100 * averageError,'%')
        avgError.append(averageError)
        x_learners.append(lr)
    # graph the errors now.
    plt.plot(x_learners, avgError)
    plt.ylabel('Average Error (k=10)')
    plt.xlabel('Learning Rate')
    plt.title('Error as a function of the learning rate')
    plt.show()

def experiment_estimators_AdaBoostRandomForest():
    avgError = []
    x_learners = []
    rf = RandomForestClassifier(n_estimators=maxLearners, max_depth = maxDepth, warm_start = False)
    for k_estimators in range(10,150,10):
        k = 10
        skf = StratifiedKFold(labels,n_folds=k)
        averageError = 0.0
        for train_index, test_index in skf:
            X_train, X_test = mfcc[:,train_index], mfcc[:,test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            adb = AdaBoostClassifier(base_estimator=rf, n_estimators=k_estimators, learning_rate=0.01)
            adb.fit(X_train.T,y_train)
            y_pred = adb.predict(X_test.T)
            error = zero_one_loss(y_pred,y_test)
            print error
            averageError += (1./k) * error
        print "Average error: %4.2f%s" % (100 * averageError,'%')
        avgError.append(averageError)
        x_learners.append(k_estimators)
    # graph the errors now.
    plt.plot(x_learners, avgError)
    plt.ylabel('Average Error (k=10)')
    plt.xlabel('Number of Estimators')
    plt.title('Error as a function of the number of estimators')
    plt.show()

def analysis_AdaBoostRandomForest():
    rf = RandomForestClassifier(n_estimators=maxLearners, max_depth = maxDepth, warm_start = False)
    k = 10
    avg_cm = np.zeros((len(genres),len(genres)))
    skf = StratifiedKFold(labels,n_folds=k)
    averageError = 0.0
    for train_index, test_index in skf:
        X_train, X_test = mfcc[:,train_index], mfcc[:,test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        adb = AdaBoostClassifier(base_estimator=rf, n_estimators=10, learning_rate=0.01)
        adb.fit(X_train.T,y_train)
        y_pred = adb.predict(X_test.T)
        error = zero_one_loss(y_pred,y_test)
        avg_cm += confusion_matrix(y_test, y_pred)
        print error
        averageError += (1./k) * error
    print "Average error: %4.2f%s" % (100 * averageError,'%')
    plot_confusion_matrix(avg_cm / k)
    plt.show()

# Neural Net used
# http://scikit-neuralnetwork.readthedocs.org/
# def experiment_convolution_net():
#     avgError = []
#     x_learners = []
#     for channels in range(1, 20, 1):
#         k = 10
#         skf = StratifiedKFold(labels,n_folds=k)
#         averageError = 0.0
#         for train_index, test_index in skf:
#             X_train, X_test = mfcc[:,train_index], mfcc[:,test_index]
#             y_train, y_test = labels[train_index], labels[test_index]
#             nn = Classifier(
#                 layers=[
#                     Layer("Linear", units=100, pieces=2),
#                     Layer("Softmax")],
#                 learning_rate=0.001,
#                 n_iter=25
#             )
#             nn.fit(X_train.T, y_train)
#             y_pred = nn.predict(X_test.T)
#             error = zero_one_loss(y_pred,y_test)
#             print error
#             averageError += (1./k) * error
#         print "Average error: %4.2f%s" % (100 * averageError,'%')
#         avgError.append(averageError)
#         x_learners.append(channels)

#     plt.plot(x_learners, avgError)
#     plt.ylabel('Average Error (k=10)')
#     plt.xlabel('Number of Channels')
#     plt.title('Error as a function of the number of channels in the CNN')
#     plt.show()

analysis_AdaBoostRandomForest()

