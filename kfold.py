from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

x, y = load_wine(return_X_y=True)
x_std = StandardScaler().fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size=0.2, random_state=0)
forest = RandomForestClassifier()
forest.fit(x_train, y_train)
predictions = forest.predict(x_test)
val_score = cross_val_score(forest, x_train, y_train, cv=5)



print(forest.score(x_train, y_train).round(3))
print(val_score.mean().round(3))
print(forest.score(x_test, y_test).round(3))
