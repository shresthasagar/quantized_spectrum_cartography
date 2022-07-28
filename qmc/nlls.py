"""
Non linear least squares (NLLS) is used to estimate the offset parameter f in the model 
y=log(f+x). 
Given the raw bin boundaries x, we want to find the transformed bin boundaries y such that the spacing between the bins is as equal as possible. 
To this end, y is chosen to take values 0,1,2...,n where n is the number of bin boundaries.

Variable raw_data stores the input bin boundaries x 
"""

import numpy as np

raw_data = [0.0, 3.219041422308777e-10, 6.34243551758118e-05, 0.0001823223865358159, 0.00036289551644586027, 0.0006664704997092485, 0.0012639077613130212, 0.00301913358271122, 0.3312782347202301]
raw_data = [0.0, 6.34243551758118e-05, 0.0001823223865358159, 0.00036289551644586027, 0.0006664704997092485, 0.0012639077613130212, 0.00301913358271122, 0.3312782347202301]
raw_data = [0.0, 8.944017748646615e-10, 2.3812383005861193e-05, 6.808515900047496e-05, 0.00012131989933550358, 0.00018234866729471833, 0.00025588355492800474, 0.00034619917278178036, 0.0004588317824527621, 0.0006049227667972445, 0.0007961964583955705, 0.0010579598601907492, 0.001441714819520712, 0.0020772861316800117, 0.003326504724100232, 0.006930550094693899, 0.27432483434677124]
raw_data = [0.0, 2.3812383005861193e-05, 6.808515900047496e-05, 0.00012131989933550358, 0.00018234866729471833, 0.00025588355492800474, 0.00034619917278178036, 0.0004588317824527621, 0.0006049227667972445, 0.0007961964583955705, 0.0010579598601907492, 0.001441714819520712, 0.0020772861316800117, 0.003326504724100232, 0.006930550094693899, 0.27432483434677124]
# raw_data =  [-23.025850296020508, -10.002398490905762, -7.980128765106201, -6.692554473876953, -1.0331487655639648]


x = np.expand_dims(np.array(raw_data), axis=1)

y = np.expand_dims(np.arange(x.shape[0]), axis=1)

def dh(x, nominal):
    return np.concatenate((1/(nominal[0,0]+x), np.ones((x.shape))), axis=1)

def h(theta, x):
    print('x', x+ theta[0,0])
    return np.log(theta[0,0] + x) + theta[1,0]

t0 = 1e-7
nominal = np.expand_dims(np.array([t0, 0]), axis=1)
H = dh(x, nominal)

for i in range(40):
    theta_ls = nominal + np.matmul(np.matmul(np.linalg.inv(np.matmul(H.T, H)), H.T), (y - h(nominal, x)))
    nominal = theta_ls.copy()
    H = dh(x, nominal)
    # print('theta is ', theta_ls)
    # print('h(theta_hat) is ', h(theta_ls, x).T)
print('theta', theta_ls)
print((h(theta_ls, x)-theta_ls[1,0]).squeeze())
