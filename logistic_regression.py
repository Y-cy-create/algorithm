from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

x, y = make_blobs(n_samples=200, n_features=2, centers=2, random_state=0)
x_std = StandardScaler().fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size=0.2, random_state=0)

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

predictions = log_reg.predict(x_test)

print(log_reg.score(x_train, y_train))
print(log_reg.score(x_test, y_test))
