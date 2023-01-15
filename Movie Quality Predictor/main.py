import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


def main():
    df = pd.read_csv('dataset.csv')
    df.drop(df[(df['rating'] <= 7) & (df['rating'] >= 5)].index, inplace=True)
    df['label'] = df.rating.apply(lambda x: 1 if x > 7 else 0)
    df.drop(columns='rating', inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(df.review.values, df.label.values, random_state=23)
    vectorizer = TfidfVectorizer(sublinear_tf=True)
    tfid_train = vectorizer.fit_transform(X_train)
    tfid_test = vectorizer.transform(X_test)

    features_num = 103  # Number of features found previously
    rounded_features = np.round(features_num, -2)
    model = LogisticRegression(solver='liblinear')

    tsvd = TruncatedSVD(n_components=rounded_features, random_state=23)
    pca_train = tsvd.fit_transform(tfid_train)
    pca_test = tsvd.transform(tfid_test)

    model.fit(pca_train, y_train)
    print(round(metrics.accuracy_score(y_test, model.predict(pca_test)), 5))
    print(round(metrics.roc_auc_score(y_test, model.predict_proba(pca_test)[:, 1]), 5))


if __name__ == '__main__':
    main()
