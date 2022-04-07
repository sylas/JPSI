from sklearn.cluster import KMeans
import numpy as np

data = [[1, 1, 1],
        [1, 1, 2],
        [8, 8, 8],
        [8, 8, 7],
        [1, 2, 1]]

print("Dane przeznaczone do klasteryzacji (wierszami):")
print(data)

NUMBER_OF_CLUSTERS = 3

results = KMeans(n_clusters=NUMBER_OF_CLUSTERS).fit_predict(np.array(data))

print("Dane po klasteryzacji:")
for i, label in enumerate(results):
    print("Dane: {}, klaster: {}".format(data[i], label))
