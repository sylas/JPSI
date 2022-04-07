import matplotlib.pyplot as plt
import numpy as np

ztable = np.random.random((100,100))

plt.imshow(ztable, cmap='hot', interpolation='nearest', aspect='auto')
plt.colorbar().set_label("Oś Z")
plt.xlabel("Oś X")
plt.ylabel("Oś Y")
plt.title("Tytuł wykresu", loc="left", pad="30")    
plt.savefig("figure.png")
plt.show()

