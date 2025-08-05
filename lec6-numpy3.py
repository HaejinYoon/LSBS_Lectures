import numpy as np
import matplotlib.pyplot as plt
# 난수 생성하여 3x3 크기의 행렬 생성
np.random.seed(2024)
img1 = np.random.rand(3, 3)
img1

plt.figure(figsize=(10, 5))  # (가로, 세로) 크기 설정
plt.imshow(img1, cmap='gray', interpolation='nearest');
plt.colorbar();
plt.show();

img_mat = np.loadtxt('./data/img_mat.csv', delimiter=',', skiprows=1)
np.min(img_mat)
np.max(img_mat)

# 행렬 값을 0과 1 사이로 변환
img_mat = img_mat / 255.0
img_mat.shape

img_mat = img_mat + 0.2
over1 = np.where(img_mat>1)
img_mat[over1] = 1
#1단계 : 전체에 0.2를 더함
#2단계: 1이상 값을 가지는 애들 -> 1로 변환

import matplotlib.pyplot as plt
# 행렬을 이미지로 변환하여 출력
plt.imshow(img_mat.transpose(), cmap='gray', interpolation='nearest');
plt.imshow(img_mat, cmap='gray', interpolation='nearest');
plt.colorbar();
plt.show();

x = np.arange(1, 11).reshape((5, 2)) * 2
y = np.arange(1, 7).reshape((2, 3))
print("원래 행렬 x:\n", x)
print("행렬 y:\n", y)

x.shape
y.shape

dot_product = x.dot(y)
dot_product
np.matmul(x, y)

mat_A = np.array( [[1, 2],
                   [4, 3]])
#(1, 2,
# 4, 3)

mat_B = np.array([[2, 1],
                 [3, 1]])
#(2, 1,
# 3, 1)

#두 행렬의 곱을 구해보세요
mat_A.dot(mat_B)
np.matmul(mat_A, mat_B)
mat_A @ mat_B

mat_A * mat_B

#역행렬
inv_A = np.linalg.inv(mat_A)
mat_A @ inv_A

# 행렬의 세계 : 1 == 단위행렬, 
# 행렬의 세계 : 역수 == 역행렬
# 
np.eye(2)
mat_A @ np.eye(2)

mat_C = np.array([[3,1],
                            [6,2]])
inv_C = np.linalg.inv(mat_C)

# 행렬 => 역행렬이 존재하는 행렬 vs. 존재하지 않는 행렬
# non-singular vs. singular
# 칼럼이 선형 독립인 경우 -> 역행렬 존재
# 칼럼이 선형 종속인 경우 -> 역행렬 X

# 고차원 행렬
# 두 개의 2x3 행렬 생성
mat1 = np.arange(1, 7).reshape(2, 3)
mat2 = np.arange(7, 13).reshape(2, 3)
# 3차원 배열로 합치기
my_array = np.array([mat1, mat2])
my_array
my_array.shape

my_array[0, :, :]
my_array[1, :, :]

my_array[:,1:, :] # [4, 5, 6],  [10, 11, 12] 접근

my_array.reshape(2, 3, 2)
my_array.reshape(-1, 3, 2)


#사진은 배열이다
#png 파일

import imageio
import numpy as np
# 이미지 읽기
jelly = imageio.imread("./data/stat.png")
print("이미지 클래스:", type(jelly))
print("이미지 차원:", jelly.shape)

import matplotlib.pyplot as plt
plt.imshow(jelly);
plt.axis('off');
plt.show();

# 흑백으로 변환
bw_jelly = np.mean(jelly[:, :, :3], axis=2) # axis는 (1094, 1794, 4)에서 기준으로 잡을 곳..
bw_jelly = np.mean(jelly[:, :, :3], axis=1)
plt.imshow(bw_jelly, cmap='gray');
plt.axis('off');
plt.show();

bw_jelly.shape

#넘파이 행렬 연산 연습
#1
mat_A = np.array([[1,2], [3,4]])
mat_B = np.array([[5,6], [7,8]])

mat_A.dot(mat_B)
mat_B.dot(mat_A)

#2
mat_A = np.array([[1, 2, 3], [4, 5, 6]])
mat_B = np.array([[7, 8], [9, 10], [11, 12]])
mat_A.dot(mat_B)
mat_A @mat_B

#3
mat_A = np.array([[2, 3], [4, 5]])
mat_I = np.array([[1, 0], [0, 1]])
mat_I = np.zeros([2, 2])
mat_A @ mat_I
mat_I @ mat_A

#4
mat_A = np.array([[1,2], [3,4]])
mat_Z = np.array([[0 ,0], [0, 0]])
mat_A @ mat_Z

#5
mat_D = np.array([[2, 0], [0, 3]])
mat_A = np.array([[4, 5], [6, 7]])
mat_A @ mat_D
mat_D @ mat_A
# D의 해당하는 행에만 영향을 준다

#6 
mat_A = np.array([[1, 2], [3, 4], [5, 6]])
mat_A = np.array([[1, 2, 5], [3, 4, 2], [5, 6, 1]])
v = np.array([[0.4], [0.6]])
v = np.array([[0.5], [0.5]])
v = np.array([[0.3], [0.3], [0.4]])
mat_A @ v

#7
mat_A = np.array([[1,2], [3,4]])
mat_B = np.array([[5,6], [7,8]])
mat_C = np.array([[9, 10], [11, 12]])
mat_T = np.stack([mat_A, mat_B], axis=0)
mat_T @ mat_C
mat_T.shape

#8
#역등(idempotent) 행렬
mat_S = np.array([[3, -6], [1 ,-2]])
mat_S @ mat_S
np.linalg.inv(mat_S)
# 각 행, 열의 합이 1이 된다

#9
mat_A = np.array([[1,2], [3,4]])
mat_B = np.array([[5,6], [7,8]])
mat_C = np.array([[9, 10], [11, 12]])

(mat_A@mat_B) @ mat_C
mat_A@(mat_B @ mat_C)
(mat_B @ mat_C) @ mat_A
(mat_C @ mat_A) @ mat_B
#결과값의 대각 합은 변하지 않는다. trace,,  mat_S.trace()
# 순서를 바꿔서 곱해도 똑같다.

#10
mat_A = np.array([[3, 2, -1], [2, -2, 4], [-1,0.5, -1]])
mat_b = np.array([[1], [-2], [0]])

x = np.linalg.inv(mat_A) @ mat_b
x
mat_A @ x