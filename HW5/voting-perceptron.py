import sys
import numpy as np
import matplotlib.pyplot as plt
import heapq

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

def weighted_majority(x, h):
    sum_c = 0
    l = len(h)
    for j in range(l):
        sum_c += h[j][0] * sign(np.dot(np.array(h[j][1]), x))
    return sign(sum_c)

def averaged_perceptron(h):
    l = len(h)
    return 0
#    sum_c = np.zeros((3, ))
#    for j in range(1, l+1):
#        sum_c += c[j] * np.array(w[j])
#    return sum_c

def predict(x, h):
    t = x.shape[0]
    z = np.zeros((t,))

    mode = sys.argv[1]

    for i in range(t):
        if mode == "a":
            z[i] = weighted_majority(x[i], h)

        elif mode == "c":
            w_ = averaged_perceptron(h)
            z[i] = sign(np.dot(w_, x[i]))
        #if i % 100 == 0:
        #    print i, z[i]
    return z

x_l = []
y_l = []
for l in sys.stdin:
    x1, x2, y = map(int, l.strip().split(" "))
    x_l.append([x1, x2])
    y_l.append(y)

total_n = len(y_l)
T = 10
print "Voting Perceptrons on ", total_n, " data"
print T, "times in total"
x_arr = np.array(x_l)
# Add a constant feature 1
x_arr = np.insert(x_arr, 2, 1, axis=1)
y_arr = np.array(y_l)

data_n = x_arr.shape[0]

x_arr_rand = np.zeros((data_n, 3))
y_arr_rand = np.zeros((data_n, ))

# Keep L different w
L = 1000000

#h = [(-1, [-1,-1,-1])]
h = []

w = [0,0,0]
c = 0

while T > 0:
    T = T - 1
    # Rand perm
    rand = np.random.permutation(data_n)
    for i in range(data_n):
        t = rand[i]
        x_arr_rand[i] = x_arr[t]
        y_arr_rand[i] = y_arr[t]
    
    for i in range(data_n):
        if sign(np.dot(np.array(w), x_arr_rand[i])) !=  y_arr_rand[i]:
            if len(h) < L:
                heapq.heappush(h, (c, w))
            else:
                heapq.heappush(h, (c, w))
                heapq.heappop(h)
            w = (np.array(w) + y_arr_rand[i] * x_arr_rand[i]).tolist()
            c = 1
        else:
            c += 1

print "h calculation finished"
print h

X = x_arr[:, :2]
Y = y_arr
hs = .03  # step size in the mesh

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, hs),
                                     np.arange(y_min, y_max, hs))

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
fig, ax = plt.subplots()
Z = predict(np.c_[xx.ravel(), yy.ravel(), np.ones(xx.ravel().shape[0])], h)

# Put the result into a color plot
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=plt.cm.Paired)
ax.axis('off')

# Plot also the training points
ax.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

ax.set_title('Perceptron')

plt.show()
