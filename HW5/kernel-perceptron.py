import sys
import numpy as np
import matplotlib.pyplot as plt

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

def rbf(x, z):
    t = 100
    return np.exp(- np.square(np.linalg.norm(x - z))/ (2*t*t))

def predict_x(x, x_arr, y_arr, a_arr):
    sum_ = 0
    for i in range(a_arr.shape[0]):
        if sys.argv[1] == "kernel":
            sum_ += a_arr[i] * y_arr[i] * np.square(1+ np.dot(x_arr[i], x)) 
        elif sys.argv[1] == "rbf":
            sum_ += a_arr[i] * y_arr[i] * rbf(x_arr[i], x)
    return sign(sum_)

def predict(x, x_arr, y_arr, a_arr):
    t = x.shape[0]
    z = np.zeros((t,))
    for i in range(t):
        z[i] = predict_x(x[i], x_arr, y_arr, a_arr)
    return z

x_l = []
y_l = []
for l in sys.stdin:
    x1, x2, y = map(int, l.strip().split(" "))
    x_l.append([x1, x2])
    y_l.append(y)

total_n = len(y_l)

T = 3000

x_arr = np.array(x_l)
# Add a constant feature 1
x_arr = np.insert(x_arr, 2, 1, axis=1)
y_arr = np.array(y_l)
a_arr = np.zeros((total_n,))

while T > 0:
    if T % 100 == 0:
        print T
    flag = 0
    for i in range(total_n):
        sum_ = 0
        for j in range(total_n):
            if sys.argv[1] == "kernel":
                sum_ += a_arr[j] * y_arr[j] * np.square(1 + np.dot(x_arr[i], x_arr[j]))
            elif sys.argv[1] == "rbf":
                sum_ += a_arr[j] * y_arr[j] * rbf(x_arr[i], x_arr[j])
        if y_arr[i] * sum_ <= 0:
            a_arr[i] += 1
            flag = 1
    if flag == 0:
        break
    T -= 1

print a_arr

X = x_arr[:, :2]
Y = y_arr
hs = .05  # step size in the mesh

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, hs),
                                     np.arange(y_min, y_max, hs))

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
fig, ax = plt.subplots()
Z = predict(np.c_[xx.ravel(), yy.ravel(), np.ones(xx.ravel().shape[0])],
            x_arr, y_arr, a_arr)

# Put the result into a color plot
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=plt.cm.Paired)
ax.axis('off')

# Plot also the training points
ax.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

ax.set_title('Perceptron')

plt.show()
