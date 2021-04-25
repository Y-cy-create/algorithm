from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
x, y = load_iris(return_X_y=True)
# print(x[0])
# print(y[0])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)
predictions = tree.predict(x_test)

print(tree.score(x_train, y_train))
print(tree.score(x_test, y_test))

