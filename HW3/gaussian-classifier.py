import numpy as np
import sys
import os
import mnist

from scipy.stats import multivariate_normal

def shuffle_data(img, label):
    m = len(img)
    n = len(img[0])
    rand_perm_list = np.random.permutation(m).tolist()
    shuffle_img = []
    shuffle_label  = []
    for i in range(m):
        shuffle_img.append(img[rand_perm_list[i]])
        shuffle_label.append(label[rand_perm_list[i]])
    return shuffle_img, shuffle_label

def read_data():
    mn = mnist.MNIST("./")
    test_img, test_label = mn.load_testing()
    train_img, train_label = mn.load_training()
    return test_img, test_label, train_img, train_label

def sort_by_class(train_img, train_label):
    list_by_class = [[] for i in range(10)]
    for i in range(len(train_label)):
        class_img = train_label[i]
        list_by_class[class_img].append(train_img[i])
    return list_by_class

# Set up the multivariate gaussian of class i, return \mu and \Sigma, m
def setup_gaussian(list_by_class, i, c=0):
    # images m*p 
    images = np.matrix(list_by_class[i])
    m = images.shape[0]
    n = images.shape[1]
    mu = np.sum(images, axis=0)/(m *1.0)
    sigma = (images - mu).T * (images - mu)
    sigma = sigma / (m*1.0)
    sigma += c * np.identity(n)
    return m, mu, sigma

def test(test_img, test_label, m_array, mu_array, sigma_array):
    error = 0
    pi_array = np.log(m_array / np.sum(m_array))
    test_n = len(test_label)
    print "Run ", test_n, " Tests"

    test_img_array = np.array(test_img)
    p_x = np.zeros((10, test_n))
    for k in range(10):
        p_x[k] = multivariate_normal.logpdf(test_img_array,
                                         mean=mu_array[k],
                                         cov=sigma_array[k])
    px_T = p_x.T + pi_array
    for i in range(test_n):
        class_id = np.argmax(px_T[i])
        if class_id != test_label[i]:
            error += 1

    return error * 100.0 / test_n

if __name__ == "__main__":
    class_n = 10
    dimen = 784
    c = float(sys.argv[1])
    test_img, test_label, train_img, train_label = read_data()

    # Separate into validation set
    #validation = 50000
    #shuffle_img, shuffle_label = shuffle_data(train_img, train_label)
    #train_img = shuffle_img[:validation]
    #train_label = shuffle_label[:validation]
    #test_img = shuffle_img[validation:]
    #test_label = shuffle_label[validation:]

    list_by_class = sort_by_class(train_img, train_label)
    #if os.path.isfile('m_array.npy'): 
    if False:
        m_array = np.load('m_array.npy')
        mu_array = np.load('mu_array.npy')
        sigma_array = np.load('sigma_array.npy')
    else:
        m_array = np.zeros((class_n))
        mu_array = np.zeros((class_n, dimen))
        sigma_array = np.zeros((class_n, dimen, dimen))
        #sigma_logdet = np.zeros((class_n))
        #sigma_inv = np.zeros((class_n, dimen, dimen))
        for i in range(10):
            m_array[i], mu_array[i], sigma_array[i] = setup_gaussian(list_by_class, i, c)
            #(sign, sigma_logdet[i]) = np.linalg.slogdet(sigma_array[i])
            #sigma_inv[i] = np.linalg.inv(sigma_array[i])
            

        #np.save('m_array.npy', m_array)
        #np.save('mu_array.npy', mu_array)
        #np.save('sigma_array.npy', sigma_array)

    err_rate = test(test_img, test_label, m_array, mu_array, sigma_array)
    print "Error rate: ", err_rate
