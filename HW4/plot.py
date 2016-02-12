import matplotlib.pyplot as plt

x1 = [2, 1, 1]
x2 = [1, 20, 5]

x3 = [4, 1, 3]
x4 = [1, 40,30]

w1 = [0, 3.128077]
w2 = [42.81853918, 0]

# Plot
plt.plot(x1, x2, 'go')
plt.plot(x3, x4, 'ro')

plt.plot(w1, w2, 'b')

# Label
plt.xlabel("x_1")
plt.ylabel("x_2")

# Axis
plt.axis([0, 6, 0, 45])

#title
plt.title("Gradient Regression")

plt.show()
