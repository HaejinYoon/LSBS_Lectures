import numpy as np

#넘파이 활용하기 연습문제
#1
a = np.array([1, 2, 3, 4, 5])
a + 5

#2
a = np.array([12, 21, 35, 48, 5])
a[::2]

#3
a = np.array([1, 22, 93, 64, 54])
a[a.argmax()]
np.max(a)

#4
a = np.array([1, 2, 3, 2, 4, 5, 4, 6])
tmp = set(a)
tmp
a = np.array(tmp)
a

np.unique(a)

#5
a = np.array([21, 31, 58])
b = np.array([24, 44, 67])
c = np.empty(a.size + b.size, dtype=a.dtype)
c[0::2] = a
c[1::2] = b
c

#6
a = np.array([1, 2, 3, 4, 5])
b = np.array([6, 7, 8, 9])

c = a[:-1] + b
c

#7
a = np.array([1, 3, 3, 2, 1, 3, 4, 2, 2, 2, 5, 6, 6, 6, 6])
a
np.unique(a)
unique, counts = np.unique(a, return_counts=True)
#np.argmax(counts)
most_frequent = unique[counts == np.max(counts)]
most_frequent

#8
a = np.array([12, 5, 18, 21, 7, 9, 30, 25, 3, 6])

multple3 = a[a%3 == 0]
multple3

#9
a = np.array([10, 20, 5, 7, 15, 30, 25, 8])
length = int(len(a)/2)
a1 = a[:length]
a2 = a[length::]

#10
a = np.array([12, 45, 8, 20, 33, 50, 19])
middle = np.median(a)
diff = np.abs(a-middle)
diff.min()
