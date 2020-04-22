from sklearn.datasets import  load_iris
from  sklearn.model_selection import train_test_split
iris_dateset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dateset.data,iris_dateset.target,random_state=0)
print(X_train)