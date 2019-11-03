import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df = pd.read_csv("pima-indians-diabetes.csv")
train,test = train_test_split(df,test_size=0.2)
X = train.loc[:,"Pregnancies":"DiabetesPedigreeFunction"]
y = train["Class"]
X_test = test.loc[:,"Pregnancies":"DiabetesPedigreeFunction"]
y_test = test["Class"]
classifier = GaussianNB()
#training
classifier.fit(X,y)
y_predicted = classifier.predict(X_test)
score = accuracy_score(y_test,y_predicted)
print( "Accuracy ",score)


# ********************OUTPUT****************************************

# chanchald@chanchald-X553SA:~$ cd Desktop
# chanchald@chanchald-X553SA:~/Desktop$ cd practicalsss
# chanchald@chanchald-X553SA:~/Desktop/practicalsss$ cd FinalDAPrac
# chanchald@chanchald-X553SA:~/Desktop/practicalsss/FinalDAPrac$ cd p2
# chanchald@chanchald-X553SA:~/Desktop/practicalsss/FinalDAPrac/p2$ python pimaanalysis.py
# Accuracy  0.7857142857142857
# chanchald@chanchald-X553SA:~/Desktop/practicalsss/FinalDAPrac/p2$ 


