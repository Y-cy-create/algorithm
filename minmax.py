from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

x, y = make_blobs(n_samples=200, n_features=2, centers=2, random_state=0)

x_std = MinMaxScaler().fit_transform(x)

plt.scatter(x_std.T[0], x_std.T[1], c=y, cmap='Dark2')
plt.grid(True)
plt.show()