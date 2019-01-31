
# SKLEARN.
from sklearn.model_selection import KFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import  SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.dummy import DummyClassifier
# NLTK.
from nltk import NaiveBayesClassifier
from nltk.classify import SklearnClassifier
from nltk.classify import accuracy
# Classes.
from Preprocesser import Preprocesser
from FeatureSetGenerator import *
from DisplayGraphs import displayGraph
# Imports.
import numpy as np
import pandas as pd


filename_output = "dataset.csv"
filename_train = "./train.tsv"
filename_test = "./test.tsv"

# Apply k-fold cross validation and return the score.
def cross_validate(classifier, data, n_split, verbose = False):
    kf = KFold(n_splits=n_split)
    sum = 0.0
    counter = 0
    for train, test in kf.split(data):
        if verbose:
            print(str(round(counter * 100 / n_split, 3)) + "%")
            counter += 1
        train_set = np.array(data)[train]
        test_set = np.array(data)[test]
        clf = classifier.train(train_set)
        sum += accuracy(clf, test_set)
    if verbose:
        print(str(round(counter * 100 / n_split, 1)) + "%")
    return round(sum / n_split, 5)

# Estimate the accuracy of the baseline approach.
def baseline():
    # Read CSV files.
    train_data = pd.read_csv(filename_train, sep='\t')
    X_raw = np.array(train_data["Phrase"])
    Y_raw = np.array(train_data["Sentiment"])

    tokenizer = ToktokTokenizer()
    all_words = []
    # Tokenize.
    for row in X_raw:
        all_words += tokenizer.tokenize(row)

    # Generate labeled_featuresets.
    print("start generating featuresets (baseline)")
    labeled_featuresets = baseline(
        X_raw,
        Y_raw,
        verbose=True
    )

    # Train the classifier.
    print("Train classifier")
    clf = SklearnClassifier(DummyClassifier()).train(labeled_featuresets)

    # Apply cross validation.
    print("Start cross validation (baseline)")
    cv_score = cross_validate(clf, labeled_featuresets, 3, True)
    print("Cross validation score: " + str(cv_score) + "%")

def generateOutput(clf, preprocesser, verbose = True):
    # Read test CSV.
    test_data = pd.read_csv(filename_test, sep='\t')
    ids = np.array(test_data["PhraseId"])
    X_raw = np.array(test_data["Phrase"])

    # Generate featuresets with no labels.
    print("start generating featuresets with no labels")
    featuresets = non_labeled(X_raw, preprocesser, verbose=True)

    # Generate the CSV output string.
    print("Generate CSV output string")
    predictions = ["PhraseId,Sentiment"]
    l = len(X_raw)
    for i in range(l):
        if (i == 0 or i % 10000 == 0) and verbose:
            print(str(round(i / l, 3)*100) + "%")
        sentiment = clf.classify(featuresets[i])
        predictions.append(str(ids[i]) + "," + str(sentiment))

    # Write to CSV.
    print("Write CSV to file")
    # Write the CVS output string to file.
    with open(filename_output, 'w') as f:
        for line in predictions:
            f.write(line)
            f.write("\n")

def main(clf, train_count=0, verbose = True):
    # Read train CSV.
    train_data = pd.read_csv(filename_train, sep='\t')

    # Get Text and Labels.
    X_raw = np.array(train_data["Phrase"])
    Y_raw = np.array(train_data["Sentiment"])
    X = X_raw
    Y = Y_raw
    if train_count != 0:
        # Use a subset of the training count.
        X = X_raw[:train_count]
        Y = Y_raw[:train_count]

    # Preprocess the data.
    print("Start Preprocess")
    preprocesser = Preprocesser(X)
    print("most common words: " + str(preprocesser.words_frequency.most_common(100)))
    if train_count != 0:
        X = X_raw[train_count:train_count*2]
        Y = Y_raw[train_count:train_count*2]

    # Construct the featuresets.
    print("Start Construct featuresets")
    labeled_featuresets = labeled(
        X,
        Y,
        preprocesser,
        verbose=True
    )

    # Construct classifier.
    print("Start training classifier")
    clf = SklearnClassifier(clf).train(labeled_featuresets)

    # cross vvalidation.
    print("Start cross-validation")
    cv_score = cross_validate(clf, labeled_featuresets, 3, True)
    print("Accuracy = " + str(cv_score))

    # Generate output
    generateOutput(clf, preprocesser)


# ***   MAIN CODE   ***

clf = LogisticRegression(solver="lbfgs", multi_class="multinomial")

#displayGraph(filename_train)
#baseline()
#main(clf, train_count=10000) # Used for limited inputs.
main(clf)
