import pandas as pd
import numpy as np

from Plotting.Decision_Region import plot_decision_region
from Model.Adaline import AdalineGD
from Model.AdalineSGD import AdalineSGD

df = pd.read_csv('iris.data', header=None)

# CLASS MAPPING

class_mapping = {label:idx for idx,label in enumerate(np.unique(df[4]))}
print class_mapping

df[4] = df[4].map(class_mapping)

y = df[4].values
X = df.iloc[:, [2,3]].values

# DATASET SPLITTING

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# STANDARDIZATION

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# PERCEPTRON

from sklearn.linear_model import Perceptron

ppn = Perceptron(n_iter=40, eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

# LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=100.0, random_state=1)
lr.fit(X_train_std, y_train)

# PREDICTION PROBABILITY

predict_proba = lr.predict_proba(X_test_std[:3, :]).argmax(axis=1)
print predict_proba

# PREDICTION - ACCURACY

from sklearn.metrics import accuracy_score

y_pred = ppn.predict(X_test_std)
ac = accuracy_score(y_test, y_pred)

print ac

# PLOTTING DECISION REGION

# plot_decision_region(X_train_std, y_train, classifier=ppn)

# ADALINE

ad = AdalineSGD(n_iter=10, eta=0.01).fit(X, y)

