from collections import Counter
import numpy as np
import pandas as pd
import pickle
from sklearn import svm, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_validate, train_test_split


def process_data_for_labels(ticker):
    hm_days = 7
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, hm_days + 1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - (df[ticker])) / df[ticker]

    df.fillna(0, inplace=True)
    return tickers, df, hm_days


def buy_sell_hold(*args):
    # *args allows to pass any number of arguments
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0


# function to map the dataframe "process_data_for_labels(ticker)" to the helper function "buy_sell_hold(*args)"
def extract_featuresets(ticker):
    tickers, df, hm_days = process_data_for_labels(ticker)

    # creating the target labels for ML
    df['{}_target'.format(ticker)] = list(
        map(buy_sell_hold, *[df['{}_{}d'.format(ticker, i)] for i in range(1, hm_days + 1)]))
    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))

    # Manage outlying values: N/As, infinite
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf, np.nan])
    df.dropna(inplace=True)

    # normalizing data to percent change
    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    return X, y, df


def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])

    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print('Accuracy', confidence)
    predictions = clf.predict(X_test)
    print('Predicted spread:', Counter(predictions))

    return confidence


do_ml('AAPL')