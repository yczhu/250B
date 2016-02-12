import numpy as np
import matplotlib.pyplot as plt
import sys

def gradient(w, y, x, n):
    num_y = y.shape[0]
    sum_t = np.zeros((1, n))
    for i in range(num_y):
        e_const = 1.0 / (1.0 + np.exp(y[i] * np.dot(x[i].reshape((1, n)), w))) 
        sum_t += y[i] * x[i] * e_const[0][0]
    m = -sum_t.reshape((1, n))
    return m.T

thres = 0.01
# Dimension after adding a feature
n = 3
eta_t = 0.005

run_type = sys.argv[1]

if run_type == "a":
    x = np.array([[2, 1], [1,20], [1,5], [4,1], [1,40],[3,30]])
    y = np.array([-1,-1,-1,1,1,1])

elif run_type == "b":
    x = np.array([[2, 0.1], [1,2], [1,0.5], [4,0.1], [1,4],[3,3]])
    y = np.array([-1,-1,-1,1,1,1])

elif run_type == "c":
    cov = [[0.5, 0], [0, 0.5]]
    mean_neg = (1, 2)
    mean_pos = (3, 3)
    x_neg = np.random.multivariate_normal(mean_neg, cov, 50)
    x_pos = np.random.multivariate_normal(mean_pos, cov, 50)
    y_l = [-1 for i in range(50)] + [1 for i in range(50)]
    y = np.array(y_l)
    x = np.array(x_neg.tolist() + x_pos.tolist())

x = np.insert(x, n-1, values = 1, axis = 1)

w_0 = np.zeros((n, 1))
w_t = w_0 
iteration = 0
while True:
    if iteration % 1000 == 0:
        print "Iter: ", iteration
    grad_L_w = gradient(w_t, y, x, n)
    if abs(np.linalg.norm(grad_L_w)) < thres:
        break
    w_t1 = w_t - eta_t * grad_L_w
    w_t = w_t1
    iteration += 1

print iteration

print w_t
print -w_t[2] / w_t[0]
print -w_t[2] / w_t[1]

w1 = [0, -w_t[2] / w_t[0]]
w2 = [-w_t[2] / w_t[1], 0]

neg = x.shape[0] / 2

x1 = [t[0] for t in x[:neg]]
x2 = [t[1] for t in x[:neg]]

x3 = [t[0] for t in x[neg:]]
x4 = [t[1] for t in x[neg:]]

# Plot
plt.plot(x1, x2, 'go')
plt.plot(x3, x4, 'ro')

plt.plot(w1, w2, 'b')

# Label
plt.xlabel("x_1")
plt.ylabel("x_2")

# Axis
x1_max = max(x[:, 0])
x2_max = max(x[:, 1]) 
plt.axis([0, x1_max + 2, 0, x2_max+2])

#title
plt.title("Gradient Regression")

plt.show()
