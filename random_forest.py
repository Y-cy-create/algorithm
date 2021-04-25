from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

x, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
forest = RandomForestClassifier()
forest.fit(x_train, y_train)
predictions = forest.predict(x_test)

print(forest.score(x_train, y_train))
print(forest.score(x_test, y_test))
