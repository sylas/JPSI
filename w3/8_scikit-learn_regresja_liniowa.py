from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

X = [[0], [0.5], [1], [1.5], [2]]
Y =  [0, 0.6, 1, 1, 2]

reg = LinearRegression().fit(X, Y)
a = reg.coef_[0]
b = reg.intercept_
print("Wzór prostej: y = {:.2f}*x + {:.2f}".format(a,b))

plt.plot(X,Y, "+")

Xreg = np.arange(0,2,0.01)
Yreg =  [a*i + b for i in Xreg] 
plt.plot(Xreg,Yreg)

plt.plot()
plt.show()

