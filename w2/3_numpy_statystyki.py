import numpy as np

zm = np.arange(100)

print("MIN:", min(zm))   
print("MAX:", max(zm))
print("ŚREDNIA:", np.mean(zm))
print("MEDIANA:", np.median(zm))
print("ZAKRES:", np.ptp(zm))
print("ODCHYLENIE STANDARDOWE:", np.std(zm))
print("WARIANCJA:", np.var(zm))
print("PERCENTYL 90%:", np.percentile(zm,90))
print("HISTOGRAM:", np.histogram(zm))

