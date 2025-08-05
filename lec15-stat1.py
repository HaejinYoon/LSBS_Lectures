import numpy as np

# 중복 없이 1부터 10까지 숫자 중 5개 무작위 추출
numbers = np.random.choice(np.arange(1, 11), size=5, replace=False)
numbers = np.random.choice(np.arange(1, 6), size=3, replace=True)
print(numbers)


#예제: 접시를 꺨 확률 2
#순이 젤리 뭉이
#29.4% 35.3%, 35.3%

import numpy as np

# 근무 확률 (P(E))
work_probs = np.array([0.294, 0.352, 0.352])  # 수니, 젤리, 뭉이

# 접시 깰 조건부 확률 (P(B|E))
break_probs = np.array([0.01, 0.02, 0.03])  # 수니, 젤리, 뭉이

# 전체 확률 공식: P(B) = sum(P(B|Ei) * P(Ei))
total_prob = np.sum(work_probs * break_probs)

print(f"접시가 깨질 확률: {total_prob:.4f} ({total_prob * 100:.2f}%)")

# 종업원 이름
names = np.array(['수니', '젤리', '뭉이'])

# 근무 확률 P(E)
work_probs = np.array([0.294, 0.352, 0.352])

# 접시 깰 조건부 확률 P(B|E)
break_probs = np.array([0.01, 0.02, 0.03])

# 두 번 깨질 확률: P(B)^2 * P(E)
numerators = break_probs * work_probs

# 전체 확률(정규화 상수)
total = np.sum(numerators)

# 조건부 확률 P(E|B and B)
posterior_probs = numerators / total

# 출력
for name, prob in zip(names, posterior_probs):
    print(f"처음 깨졌을 때, {name}가 일하고 있을 확률: {prob:.4f} ({prob*100:.2f}%)")

# 근무 확률 P(E)
work_probs = np.array([0.294, 0.352, 0.352])

# 접시 깰 조건부 확률 P(B|E)
break_probs = np.array([0.01, 0.02, 0.03])

# 두 번 깨질 확률: P(B)^2 * P(E)
numerators = break_probs * work_probs

# 전체 확률(정규화 상수)
total = np.sum(numerators)

# 조건부 확률 P(E|B and B)
posterior_probs = numerators / total

# 출력
for name, prob in zip(names, posterior_probs):
    print(f"처음 깨졌을 때, {name}가 일하고 있을 확률: {prob:.4f} ({prob*100:.2f}%)")

# 종업원 이름
names = np.array(['수니', '젤리', '뭉이'])

# 근무 확률 P(E)
work_probs = np.array([0.5, 0.3, 0.2])

# 접시 깰 조건부 확률 P(B|E)
break_probs = np.array([0.01, 0.02, 0.03])

# 두 번 깨질 확률: P(B)^2 * P(E)
numerators = (break_probs**2) * work_probs

# 전체 확률(정규화 상수)
total = np.sum(numerators)

# 조건부 확률 P(E|B and B)
posterior_probs_twice = numerators / total

# 출력
for name, prob in zip(names, posterior_probs_twice):
    print(f"두 번 깨졌을 때, {name}가 일하고 있을 확률: {prob:.4f} ({prob*100:.2f}%)")

#베이즈정리 연습문제
# 문제 1 – 회사 직원의 건강 문제와 흡연 확률
# 한 회사에서 무작위로 선택된 직원이 건강 문제가 있을 확률은 0.25입니다. 건강 문제가 있는 직원은 건강 문제가 없는 직원보다 흡연자일 확률이 두 배 높습니다.

# 직원이 흡연자라는 사실을 알았을 때, 그가 건강 문제를 가지고 있을 확률을 계산하십시오.

prob = 0.25 #문제있을 확률
nprob = 0.75 #없을 확률

import numpy as np

# 확률 설정
P_G = 0.25             # 건강 문제 있는 확률
P_notG = 1 - P_G       # 건강 문제 없는 확률

# 건강 문제 없는 사람이 흡연자일 확률을 x라고 둠
x = 1  # 상대적인 값으로 설정해도 상관없음 (비율로만 계산하므로)
P_S_given_notG = x
P_S_given_G = 2 * x    # 조건: 건강 문제 있으면 두 배

# 전체 흡연자일 확률 P(S)
P_S = P_S_given_G * P_G + P_S_given_notG * P_notG
# => P_S = 2x * 0.25 + x * 0.75 = 1.25x

# 베이즈 정리: P(G|S) = (P(S|G) * P(G)) / P(S)
P_G_given_S = (P_S_given_G * P_G) / P_S

print(f"흡연자일 때 건강 문제를 가질 확률: {P_G_given_S:.4f} ({P_G_given_S * 100:.1f}%)")



#2
import numpy as np

# 버전 이름 (기타 제외)
versions = np.array(['2022', '2021', '2020'])

# 전체 비율
version_probs = np.array([0.16, 0.18, 0.20])

# 버그 발생 확률
bug_probs = np.array([0.05, 0.02, 0.03])

# 베이즈 분자: P(B|V) * P(V)
numerators = bug_probs * version_probs

# 전체 확률 (정규화 상수)
denominator = np.sum(numerators)

# 조건부 확률: P(V|B)
posterior = numerators / denominator

# 2022의 인덱스는 0
p_2022_given_bug = posterior[0]

print(f"버그 발생 시 2022 버전일 확률: {p_2022_given_bug:.3f} ({p_2022_given_bug*100:.1f}%)")


#X: 동전을 두 번 던져서 나온 앞면의 수
#X를 시각화

import matplotlib.pyplot as plt

# X값 (앞면 수)과 확률
x_vals = [0, 1, 2]
probs = [0.25, 0.5, 0.25]

# 막대그래프
plt.bar(x_vals, probs, color='skyblue', edgecolor='black')

# 라벨 및 제목
plt.xticks(x_vals)
plt.xlabel('앞면의 수 (X)')
plt.ylabel('확률 P(X)')
plt.title('동전 두 번 던졌을 때 앞면의 수 확률분포')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, 1)

# 각 확률 표시
for x, p in zip(x_vals, probs):
    plt.text(x, p + 0.02, f'{p:.2f}', ha='center')

plt.show()

#X= 동전 두번 던져서 나온 앞면 수
#P({앞}) = 0,4
#X를 시각화 하면?
#P(X=0) = 0.6 X 0.6 = 0.36
#P(X=2) = 04 X 0.4 = 0.16
#P(X=1) = 1- 0.52 = 0.48
#HT , TH = 0.4 X 0.6 + 0.6 X 0.4