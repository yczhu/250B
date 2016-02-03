import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import mnist
import sys

mn = mnist.MNIST("./")
test_img, test_label = mn.load_testing()

i = int(sys.argv[1])
img = np.array(test_img[i])
img = img.reshape((28,28))
plt.imshow(img)
plt.show()
