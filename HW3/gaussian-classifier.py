import numpy as np
import sys
import os
import mnist

from scipy.stats import multivariate_normal

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
#    sigma = np.zeros((n, n))
#    for j in range(m):
#        sigma += np.dot(np.reshape((images[j] - mu), (n, 1)),
#                        np.reshape((images[j] - mu), (1, n)))
    sigma = sigma / (m*1.0)
    sigma += c * np.identity(n)
    return m, mu, sigma

def test(test_img, test_label, m_array, mu_array, sigma_array):
    error = 0
    pi_array = m_array / np.sum(m_array)
    #test_n = len(test_label)
    test_n = 100
    for i in range(test_n):
        if i % 10 == 0:
            print "Run Test ", i, " / ", test_n
        x = test_img[i]
        max_ = - sys.maxint
        class_id = -1
        for k in range(10):
            p_x = multivariate_normal.logpdf(x,
                                          mean=mu_array[k],
                                          cov=sigma_array[k])
            if p_x * pi_array[k] > max_:
                max_ = p_x * pi_array[k]
                class_id = k
        #print class_id
        if  class_id != test_label[i]:
            error += 1

    return error * 100.0 / test_n

if __name__ == "__main__":
    class_n = 10
    dimen = 784
    test_img, test_label, train_img, train_label = read_data()
    list_by_class = sort_by_class(train_img, train_label)
    if os.path.isfile('m_array.npy'): 
        m_array = np.load('m_array.npy')
        mu_array = np.load('mu_array.npy')
        sigma_array = np.load('sigma_array.npy')
    else:
        m_array = np.zeros((class_n))
        mu_array = np.zeros((class_n, dimen))
        sigma_array = np.zeros((class_n, dimen, dimen))
        sigma_logdet = np.zeros((class_n))
        #sigma_inv = np.zeros((class_n, dimen, dimen))
        for i in range(10):
            m_array[i], mu_array[i], sigma_array[i] = setup_gaussian(list_by_class, i, 1.0)
            (sign, sigma_logdet[i]) = np.linalg.slogdet(sigma_array[i])
            #sigma_inv[i] = np.linalg.inv(sigma_array[i])
            

        np.save('m_array.npy', m_array)
        np.save('mu_array.npy', mu_array)
        np.save('sigma_array.npy', sigma_array)
        np.save('sigma_logdet.npy', sigma_logdet)

    err_rate = test(test_img, test_label, m_array, mu_array, sigma_array)
    print "Error rate: ", err_rate
