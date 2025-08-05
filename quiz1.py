import seaborn as sns
import pandas as pd
import numpy as np
df = sns.load_dataset('titanic')
df = df.dropna()
df

df[df["sex"] == "male"]["age"].sum() - df[df["sex"] == "female"]["age"].sum()
df[(df["age"].between(40, 49)) & (df["sex"] == "male")]["fare"].mean()


np.random.seed(2025)
array_2d = np.random.randint(1, 13, 200).reshape((50, 4))
array_2d[:4,:]
array_2d

np.where(array_2d.mean(axis=1).argmax())

array_2d.mean(axis=1).argmax()
array_2d[44].mean()

result = 0
for i in array_2d:
    print(i)
    max = i.max()
    min = i.min()
    result += (max - min)
    print(result)
    
    
np.array(array_2d[:, :].max())


array_2d[0].min()

import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.dot(a, b))

import numpy as np
A = np.array([[3, 6, 9], [12, 15, 18], [21, 24, 27]])
B = A[:, [0, 2]] - A[:, 1:]
print(B.shape)

import numpy as np
lst = [[1, 2, 3], [4, 5, 6]]
arr = np.array(lst)
print(lst[0][1], arr[0, 1])