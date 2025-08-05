import numpy as np

np.arange(1, 10, -.5)
np.linspace(1, 10, num=3, endpoint=False)

repeated_vals = np.repeat(8, 4)
np.repeat([1,3,5], 4)
np.tile([1,3,5],4)

arr = np.array([[1,2,],
                     [3,4]])

np.repeat(arr, 3, axis=0)
np.repeat(arr, 3, axis=1)

np.repeat([1, 2, 4], repeats=[1,2,3])

#벡터의 길이, 모양, 사이즈
vec_a = np.array([2,1,4])
len(vec_a)
vec_a.shape
vec_a.size

#행렬의 길이, 모양, 사이즈
len(arr)
arr.shape
arr.size

#브로드캐스팅
a = np.array([1, 2, 3, 4])
b = np.array([1, 2])

a + np.tile(b,2)
a + np.repeat(b,2)

# 2차원 배열 생성
matrix = np.array([[ 0.0,  0.0,  0.0],
                   [10.0, 10.0, 10.0],
                   [20.0, 20.0, 20.0],
                   [30.0, 30.0, 30.0]])
matrix.shape
# 1차원 배열 생성
vector = np.array([1.0, 2.0, 3.0])
vector = np.array([1.0, 2.0, 3.0, 4.0]).reshape(4, 1)
vector.shape
matrix + vector

matrix
#3행 4열로 바꾸고싶다
matrix.reshape(3,4)
matrix.reshape(-1).shape # -1은 열수는 알아서 계산해줘의 뜻
np.linspace(0, 100, num=20).reshape(4,5)

#벡터 인덱싱
vec_a = np.arange(20)
vec_a

vec_a[1:4]
vec_a[:7:2]

a = np.array([1, 5, 7, 8, 10])
a<7
result = np.where(a < 7)[0]
result[1]


# 두 개의 벡터를 합쳐 행렬 생성
matrix = np.column_stack((np.arange(1, 5),
                          np.arange(12, 16)))
matrix

# 2행 2열 빈 행렬 생성
y = np.zeros((2, 2))
print("빈 행렬 y:\n", y)

# 1부터 4까지의 수로 채운 2행 2열 행렬 생성
y = np.arange(1, 5).reshape(2, 2)
print("1부터 4까지의 수로 채운 행렬 y:\n", y)

# 가로 방향으로 채우기 (기본값)
y = np.arange(1, 5).reshape((2, 2), order='F')
print("가로 방향으로 채운 행렬 y:\n", y)

# 1부터 10까지의 수에 2를 곱한 값으로 5행 2열의 행렬 생성
x = np.arange(1, 11).reshape((5, 2)) * 2
x[2, 0]
x[3, 1]
x[2:4, 0]
x[1:4 ]
x[[1, 3, 4]]
x[[1,2,3], [0, 1, 1]]
x[2:4, 0]
x[2:4, [0]]

vec_a[[1,3,4]]

x[x > 10]
x[x[:,0] > 10, :]

# Q 두 번째열(기말고사) 점수가 10점 이하인 학생들 데이터를 필터링하면?
x[x[:,1] < 10, :]

np.random.seed(2025)
vec_a = np.random.choice(np.arange(1, 101), size=3000, replace=True)
mat_a = vec_a.reshape(-1, 2)
mat_a

#Q1. 중간고사 평균과 기말고사 평군은?
mid_avg = sum(mat_a[:,0])/len(mat_a) # mean() 사용
fin_avg = sum(mat_a[:,1])/len(mat_a)
mat_a.mean(axis=0)

#Q2. 중간고사 성적이 50이상인 학생들의 데이터를 걸러네보세요
len(mat_a[mat_a[:, 0]>=50, :])

#Q3. 그 학생들의 기말고사 성적 평균은?
over50 = mat_a[mat_a[:, 0]>=50, :]
fin_avg_over50 = sum(over50[:,1])/len(over50)

mat_a[mat_a[:, 0]>=50, 1].mean()

#Q4. 중간고사 최고점을 맞은 학생의 기말고사 성적은?
mid_max = mat_a[:,0].argmax()
mat_a[mid_max, 1]

mid_score=mat_a[:,0]
fin_score=mat_a[:,1]
mat_a[mid_score == max(mid_score), 1]

#Q5. 중간고사 성적이 평균보다 높은 학생들의 기말고사 성적 평균은?
over_mid_avgs = mat_a[mat_a[:,0]>mid_avg]

mat_a[mid_score > mid_avg, 1].mean()

#Q6.중간고사 대비 기말고사 성적이 향상된 학생들은 몇 명?
sum(mid_score < fin_score)

#Q7반대로 성적이 떨어진 학생들은 어디에 위치해 있나요? 학번(인덱스) 정보
std_index = np.where(mid_score > fin_score)[0]
std_index
mat_a[std_index]








