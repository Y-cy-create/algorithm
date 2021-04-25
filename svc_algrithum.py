from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC

x, y = make_moons(n_samples=500, noise=0.15, random_state=0)
x_std = StandardScaler().fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size=0.2, random_state=0)

linear_svm = LinearSVC()
linear_svm.fit(x_train, y_train)
predictions = linear_svm.predict(x_test)

svc = SVC()
svc.fit(x_train, y_train)
predictions = svc.predict(x_test)

print(linear_svm.score(x_train, y_train))
print(linear_svm.score(x_test, y_test))
print(svc.score(x_train, y_train))
print(svc.score(x_test, y_test))

