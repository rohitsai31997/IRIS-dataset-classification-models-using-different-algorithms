import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv(r"C:\Users\rohit\Desktop\Datasets\Iris.csv")

X = df.iloc[:, 1:5]
y = df.iloc[:, 5]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
print("There are {} samples in the training dataset and {} samples in the test set".format(X_train.shape, X_test.shape))

#Scaling the data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# X_train_std and X_test_std are the scaled datasets to be used in the algorithms

#Model 1 - SVC (Support Vector Classification)
from sklearn.svm import SVC

svm = SVC(kernel = 'rbf', random_state=0, gamma=10, C = 1.0)
svm.fit(X_train_std, y_train)
print('The accuracy of the SVM classifier on training data is {:.2f}'.format(svm.score(X_train_std, y_train)))
print('The accuracy of the SVM classifier on test data is {:.2f}'.format(svm.score(X_test_std, y_test)))

#Model 2 - Applying KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7, p=2, metric = 'minkowski')
knn.fit(X_train_std, y_train)
print('The accuracy of the Knn classifier on training data is {:.2f}'.format(knn.score(X_train_std, y_train)))
print('The accuracy of the Knn classifier on test data is {:.2f}'.format(knn.score(X_test_std, y_test)))

#Model 3 - Applying XGBoost
import xgboost as xgb

xgb_clf = xgb.XGBClassifier()
xgb_clg = xgb_clf.fit(X_train_std, y_train)
print('The accuracy of the XGBoost classifier on training data is {:.2f}'.format(xgb_clf.score(X_train_std, y_train)))
print('The accuracy of the XGBoost classifier on test data is {:.2f}'.format(xgb_clf.score(X_test_std, y_test)))

#Model 4 - Applying Decision Trees
from sklearn import tree

decision_tree = tree.DecisionTreeClassifier(criterion="gini")
decision_tree.fit(X_train_std, y_train)
print('The accuracy of the Decision Tree classifier on training data is {:.2f}'.format(decision_tree.score(X_train_std, y_train)))
print('The accuracy of the Decision Tree classifier on test data is {:.2f}'.format(decision_tree.score(X_test_std, y_test)))

#Model 5 - Applying Random Forest
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier()
random_forest.fit(X_train_std, y_train)
print('The accuracy of the Random Forest classifier on training data is {:.2f}'.format(random_forest.score(X_train_std, y_train)))
print('The accuracy of the Random Forest classifier on test data is {:.2f}'.format(random_forest.score(X_test_std, y_test)))
