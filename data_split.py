from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

x, y = make_blobs(n_samples=200, n_features=2, centers=2, random_state=0)
x_std = StandardScaler().fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size=0.2, random_state=0)
print(x)
print(y)
print(x.shape)
print(x_train.shape)
print(x_test.shape)
print(y.shape)
print(y_train.shape)
print(y_test.shape)