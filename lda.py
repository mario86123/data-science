import numpy as np
# import matplotlib.pyplot as plt

a = np.array([[5, 3], [3, 5], [3, 4], [4, 5], [4, 7], [5, 6]])
b = np.array([[9, 10], [7, 7], [8, 5], [8, 8], [7, 2], [10, 8]])
# a = np.array([[4, 2], [2, 4], [2, 3], [3, 6], [4, 4]])
# b = np.array([[9, 10], [6, 8], [9, 5], [8, 7], [10, 8]])
# a = np.array([[4, 1], [2, 4], [2, 3], [3, 6], [4, 4]])
# b = np.array([[9, 10], [6, 8], [9, 5], [8, 7], [10, 8]])

mean_a = np.mean(a, axis=0)
mean_b = np.mean(b, axis=0)
# print(mean_a)
# print(mean_b)

a_centered = a - mean_a
b_centered = b - mean_b

s_1 = np.zeros((2, 2))
for i in range(a.shape[0]):
    tmp_a = a_centered[i].reshape(2, 1)
    s_1 += np.matmul(tmp_a, tmp_a.T)

s_1 /= (a.shape[0] - 1)
print(s_1)

s_2 = np.zeros((2, 2))
for i in range(b.shape[0]):
    tmp_b = b_centered[i].reshape(2, 1)
    s_2 += np.matmul(tmp_b, tmp_b.T)
    
s_2 /= (b.shape[0] - 1)
print(s_2)

s_w = s_1 + s_2

tmp = (mean_a - mean_b).reshape(2, 1)
s_b = np.matmul(tmp, tmp.T)
print(s_b)
print(s_w)

fld = np.matmul(np.linalg.inv(s_w), s_b)
print(fld)
e_val, e_vec = np.linalg.eig(fld)
print(e_val)
print(e_vec)

print(e_val[0])
print(e_vec.T[0])
# plt.plot(a[:, 0], a[:, 1], 'o', color='blue')
# plt.plot(b[:, 0], a[:, 1], 'x', color='red')
# plt.show()