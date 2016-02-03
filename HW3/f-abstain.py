import numpy as np
import sys
import matplotlib.pyplot as plt

p_x = np.load("p_x.npy")

assert p_x.shape == (10000, 10)

p_x_sort = np.sort(p_x)
p_x_diff = p_x_sort[:,9] - p_x_sort[:,8]

p_x_diff_sort = np.sort(p_x_diff)
f = float(sys.argv[1])
print "Threshold: ", p_x_diff_sort[int(f * 10000)-1]

f = [0, 0.05, 0.1, 0.15, 0.2]
abstain = [0, 441, 923, 1426, 1931]
error = [4.37, 2.678, 1.6415, 1.143, 0.868]

plt.plot(f, error, 'b')
#plt.plot(f, abstain, 'g')
plt.show()
