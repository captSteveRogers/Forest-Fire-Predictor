# Importing the libraries
import numpy as np
import pandas as pd
import pickle

# Getting the dataset
df = pd.read_csv('Forest_fire.csv')
X = df.iloc[:, [1,2,3]].values
y = df.iloc[:, [4]].values

# splitting into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Classifier:
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

classifier.fit(X_train, y_train)

ipt=[int(x) for x in "45 32 60".split(' ')]
final=[np.array(ipt)]

b = classifier.predict_proba(final)

# emit through pickle
pickle.dump(classifier, open('model.pkl','wb'))
model = pickle.load(open('model.pkl', 'rb'))    