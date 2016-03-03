import numpy as np
from sklearn import manifold
from matplotlib import pyplot as plt

def draw_scatter(coords, arr_label):
    fig, ax = plt.subplots()
    ax.scatter(coords[0: N_cities, 0], coords[0: N_cities, 1])
    for i, txt in enumerate(arr_label):
        ax.annotate(txt, (coords[i][0], coords[i][1]))
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.title('City distance embedding to 2-d')
    plt.show()

if __name__ == "__main__":
    N_cities = 10
    count = 0
    X = np.zeros((N_cities, N_cities))
    list_cities = []

    with open('distances.txt', 'r') as f:
        for line in f:
            X[count] = np.array(map(int, line.strip().split(',')))
            count += 1

    with open('cities.txt', 'r') as f:
        for line in f:
            list_cities.append(line.strip())

    mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=3)
    coords = mds.fit_transform(X)
    #coords[:,[0, 1]] = coords[:,[1, 0]]

    draw_scatter(coords, list_cities)
