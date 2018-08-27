"""
Detects 5's or not 5's from the MNIST data set.
Evaluates the results using ROC and PR curves and produces a Confusion Matrix.
"""

from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, \
    precision_recall_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import matplotlib.pyplot as plt
import numpy as np


plot_image = False
Invoke_Self_Cross_validation = False
Invoke_CVS = False
# Get MNIST data set
mnist = fetch_mldata('MNIST original')

# Create Training and Test Data
X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000], y[:60000], y[60000:]
# Shuffle the training set
shuffleIndex = np.random.permutation(60000)
X_train, y_train = X_train[shuffleIndex], y_train[shuffleIndex]


#Create target vector for classification task.
#Is the digit a 5 or not a 5?
y_train_5 = (y_train ==5)
y_test_5  = (y_test == 5)


# Create a Stochastic Gradient Descent (SGD) classifier.
sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=0.1)
# train it on the training set
sgd_clf.fit(X_train, y_train_5)

# Run through 100 instances to see the classification in action.
# Enable plot_image to see it happening!
if (plot_image):
    for i in range(100):
        # Pick a random entry from the MNIST training set
        ran = np.random.randint(0, len(X_train))
        #  print(y_train[ran])
        some_digit = X_train[ran]
        # reshape it for displaying as an image

        some_digit_image = some_digit.reshape(28,28)
        plt.imshow(some_digit_image, cmap=plt.cm.binary)
        plt.draw()
        plt.pause(.1)
        plt.clf()
        # Get the SGD prediction of 5 or not 5
        prediction = sgd_clf.predict([some_digit])
        print(prediction)

# Measure Accuracy using cross-validation
# StratifiedKFold class performs stratified sampling and produces 'folds' which contain
# representative ratios of each sampling layer.
if (Invoke_Self_Cross_validation):
    skfolds = StratifiedKFold(n_splits=3, random_state=42)
    for train_index, test_index in skfolds.split(X_train, y_train_5):
        # clone the classifier sgd_clf
        clone_clf = clone(sgd_clf)
        # create stratified samples from each training data/target and test data/target (folds)
        X_train_folds = X_train[train_index]
        y_train_folds = y_train_5[train_index]
        X_test_fold = X_train[test_index]
        y_test_fold = y_train_5[test_index]
        # train the cloned classifier on the folds
        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        # Score the correct predicitions
        n_correct = sum(y_pred == y_test_fold)
        print(n_correct/len(y_pred))

# Or using sklearn cross_val_score()...
# WARNING: Accuracy isn't the preferred measure as it can be skewed by relative abundances in the data.
if (Invoke_CVS):
    CVS = cross_val_score(sgd_clf, X_train, y_train, y_train_5, cv=3, scoring="accuracy")
    print('Fold #1 : %.2f %%     Fold #3 : %.2f %%   Fold #3 : .%.2f %% ' %(100*CVS[0], 100*CVS[1], 100*CVS[2]))

# Produce a Confusion Matrix
# cross_val_predict returns the predictions made on each test fold.
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
# pass the predictions and the targets to the confusion matrix class.
labels = ['Is a 5', 'Not a 5']
CM = confusion_matrix(y_train_5, y_train_pred)
fig = plt.figure(1)
ax = fig.add_subplot(111)
cax = ax.matshow(CM, cmap=plt.cm.hot, vmin=0, vmax=60000)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# Print precision score and recall score
# Precision = TP/(TP+FP)  Ratio of true positives and all positives
precisionScore = precision_score(y_train_5, y_train_pred)
print('A high precision is required in cases where a false positive could be very bad (eg. video classifiers on YouTube)')
print('Precision Score : %.2f%% ' % (100.0*precisionScore))
# Recall    = TP/(TP+FN)  True Positive Rate (TPR)  Ratio of True positives and
# everything that should have been a true positive.
recallScore = recall_score(y_train_5, y_train_pred)
print('A high recall score is good for cases where precision isn\'t vital, but sensitivity needs to be high.')
print('Recall Score : %.2f%% ' % (100.0*recallScore))
# F1 score combines Precision & Recall. Gives more weight to low abundance values.
F1Score = f1_score(y_train_5, y_train_pred)
print('F1 score favors classifiers that have a similar precision & recall (not always what you want!)')
print('Recall F1 Score : %.2f%% ' % (100.0*F1Score))

# Produce a Precision vs Recall Curve
# FPR = False Positive Rate
# TPR = True Positive Rate
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method='decision_function')

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
fig2 = plt.figure(2)
plt.subplot(2,1,1)
plt.plot(recalls, precisions, 'r', linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision vs Recall curve')

plt.subplot(2,1,2)
plt.plot(thresholds, recalls[:-1], 'k-', linewidth=2, label='recall')
plt.plot(thresholds, precisions[:-1], 'r--', linewidth=2, label='precision')
plt.legend(loc='best')

# Produce a Receiver Operating Curve (ROC)
fpr, tpr , thresholds = roc_curve(y_train_5, y_scores)
fig3 = plt.figure(3)
plt.plot(fpr, tpr, linewidth=2,label="SGD")
plt.plot([0,1], [0,1], 'k--')
plt.axis([0,1,0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# Calculate the Area Under the Curve (AUC)
AUC = roc_auc_score(y_train_5, y_scores)
plt.text(0.6,0.4, ('SGD AUC: %.2f%%' % (100*AUC)) )


# Create a ROC curve using the RandomForestClassifier instead of
# the SGDClassified.
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
# To plot a ROC curve, we need scores, not probabilities.
y_scores_forest = y_probas_forest[:,1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)
plt.plot(fpr_forest, tpr_forest, "b:", label="Random Forest")
plt.legend(loc='best')
AUC_forest = roc_auc_score(y_train_5, y_scores_forest)
plt.text(0.6,0.45, ('Random Forest AUC: %.2f%%' % (100*AUC_forest)) )

plt.show()