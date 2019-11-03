import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("pima-indians-diabetes.csv")
# train,test = train-test-split(df,test_size=0.2)
train,test = train_test_split(df,test_size=0.2)
X = train.loc[:,"Pregnancies":"DiabetesPedigreeFunction"]
Y = train["Class"]
X_test = test.loc[:,"Pregnancies":"DiabetesPedigreeFunction"]
Y_test = test["Class"]
classifier = GaussianNB()
classifier.fit(X,Y)
y_predicted = classifier.predict(X_test)
# y_predicted = classifier.predict(X-test)
score = accuracy_score(Y_test,y_predicted)
print("Accuracy",score)