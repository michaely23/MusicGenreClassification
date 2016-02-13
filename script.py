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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import zero_one_loss

# util functions
# plot_confusion_matrix from 
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
genres = ['Blues', 'Classical', 'Country', 'Disco', 'Hiphop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
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


analysis_ideal_random_forest()
