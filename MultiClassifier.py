from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


# Chose which classifier to run demo with
SGD = False
forest = False

# Get MNIST data set
mnist = fetch_mldata('MNIST original')

# Create Training and Test Data
X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000], y[:60000], y[60000:]
# Shuffle the training set
shuffleIndex = np.random.permutation(60000)
X_train, y_train = X_train[shuffleIndex], y_train[shuffleIndex]

# Create a Stochastic Gradient Descent (SGD) classifier.
sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=0.1)
# train it on the training set
sgd_clf.fit(X_train, y_train)

# See if the prediction of the number is correct using the SGD classifier.
if(SGD):
    for i in range(100):
        ran = np.random.randint(0, len(X_train))
        some_digit = X_train[ran]
        thisDigit = sgd_clf.predict([some_digit])
        print('Prediction : %d.   Actual : %d ' %(thisDigit, y_train[ran] ))
        # Show incorrectly classified result
        if (thisDigit != y_train[ran]):
            digit_image = some_digit.reshape(28,28)
            plt.imshow(digit_image)
            plt.title('Predicted as %d. Should be %d' % (thisDigit, y_train[ran]))
            plt.draw()
            plt.pause(5)

# Or train a random forest classifier instead.
forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(X_train, y_train)
# See if the prediction of the number is correct using the RF classifier.
if(forest):
    for i in range(100):
        ran = np.random.randint(0, len(X_train))
        some_digit = X_train[ran]
        thisDigit = forest_clf.predict([some_digit])
        print('Prediction : %d.   Actual : %d ' %(thisDigit, y_train[ran] ))
        #  Print out the multiclassifier probabilities.
        print(forest_clf.predict_proba([some_digit]))
        # Show incorrectly classified result
        if (thisDigit != y_train[ran]):
            digit_image = some_digit.reshape(28,28)
            plt.imshow(digit_image)
            plt.title('Predicted as %d. Should be %d' % (thisDigit, y_train[ran]))
            plt.draw()
            plt.pause(5)



