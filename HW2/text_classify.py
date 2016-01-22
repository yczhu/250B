import numpy as np
import sys
import os

def read_data(path, file_name):
    if os.path.isfile(file_name + '.npy'):
        return np.load(file_name + '.npy')
    with open(os.path.join(path, file_name), 'r') as f:
        data_list = []
        for line in f:
            doc_id, word_id, count = map(int, line.split(' '))
            data_list.append([doc_id, word_id, count])
        data_array = np.array(data_list)
        np.save(file_name + '.npy', data_array)
    return data_array

def read_label(path, file_name):
    if os.path.isfile(file_name+'.npy') and os.path.isfile('pi.npy'):
        return np.load(file_name + '.npy'), np.load('pi.npy')
    with open(os.path.join(path, file_name), 'r') as f:
        label_list = [0]
        classify = [0 for i in range(21)]
        for line in f:
            label_list.append(int(line))
            classify[int(line)] += 1
        classify = map(lambda x: 1.0 * x / len(label_list), classify)
        label_array = np.array(label_list)
        pi = np.array(classify)
        pi[0] = 1.0
        pi = np.log2(pi)
        np.save(file_name + '.npy', label_array)
        np.save('pi.npy', pi)
    return label_array, pi

def setup_multinomial_model(label, data):
    if os.path.isfile('multinomial.npy'):
        return np.load('multinomial.npy')
    m = np.zeros((21, 61189))
    len_data = data.shape[0]
    for i in range(len_data):
        doc_id = data[i][0]
        word_id = data[i][1]
        count = data[i][2]
        classify = label[doc_id]
        m[classify][word_id] += count

    # Remove stop words
    stop_word = {12:"of", 23:"and",139:"an",978:"am",297:"at",51:"but",52:"with",33:"to",48:"on",27:"are",29:"the",72:"can",1367:"else",81:"for",301:"he",389:"she",99:"so"}
    for k in stop_word:
        m[:, k] = 0.0
    m += 1
    m[:,0] = 0.0
    s = np.sum(m, axis = 1)
    s_trans = np.transpose([s])
    m = m / s_trans
    m[:,0] = 1.0
    m = np.log2(m)

    np.save('multinomial.npy', m)
    return m

def naive_bayes(m, pi, test_data, test_label):
    len_test_data = test_data.shape[0]
    number_doc_plus_1 = len(test_label)
    test_m = np.zeros((number_doc_plus_1, 61189))
    for i in range(len_test_data):
        doc_id = test_data[i][0]
        word_id = test_data[i][1]
        count = test_data[i][2]
        test_m[doc_id][word_id] += count

    # log(1+f)
    test_m = np.log2(1+test_m)
    error = 0
    for i in range(1, number_doc_plus_1):
        cur_doc = test_m[i]
        cur_s = np.sum(cur_doc * m, axis = 1)
        final = cur_s + pi
        final = final[1:]
        label = np.argmax(final) + 1
        if label != test_label[i]:
            error += 1
    return error * 100.0 / (number_doc_plus_1 - 1)

if __name__ == "__main__":
    path = '20news-bydate/matlab/'
    label_array, pi = read_label(path, 'train.label')
    data_array = read_data(path, 'train.data')
    m = setup_multinomial_model(label_array, data_array)


    test_label, _ = read_label(path, 'test.label')
    test_data = read_data(path, 'test.data')

    err = naive_bayes(m, pi, test_data, test_label)
    print "Error Rate: ", err
