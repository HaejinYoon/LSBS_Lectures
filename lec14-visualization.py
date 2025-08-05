import matplotlib.pyplot as plt

plt.plot([4, 1, 3, 2], marker='o', linestyle='None') #marker='x' '^' 's'
plt.ylabel('Some Numbers')
plt.show()


import numpy as np
# x-y plot (산점도)
plt.plot(np.arange(10), np.arange(10), marker='o', linestyle='None')
plt.show()


import pandas as pd

df = pd.read_csv('./data/penguins.csv')
df.info()

plt.plot(df["bill_length_mm"], df['bill_depth_mm'], marker='o', linestyle='None', color = 'red')
plt.xlabel('bill length (mm)')
plt.ylabel('bill depth (mm)')

plt.show()

# 100 - 녹색, 244 - 빨간색
x = np.repeat("green", 100)
y = np.repeat("red", 244)
my_color = np.concatenate([x, y])

df['species']
# 아델리면 'red'
# 친스트랩 'blue'
# 켄터면 'green'
# 색깔벡터 만들어 보기
color = []
for peng in df['species'] :
    if(peng == 'Adelie'):
        color.append('red')
    elif(peng == 'Chinstrap'):
        color.append('blue')
    else:
        color.append('green')
color

color_map = {
    "Adelie" : "red",
    "Chinstrap" : "blue",
    "Gentoo" : "green",
}

color_vector = df['species'].map(color_map)
color_vector

plt.scatter(df["bill_length_mm"], df['bill_depth_mm'],
            #c = my_color
            c = color
            )
plt.xlabel('bill length (mm)')
plt.ylabel('bill depth (mm)')
plt.show()

x = np.repeat(0, 100)
y = np.repeat(1, 144)
z = np.repeat(2, 100)
my_color = np.concatenate([x, y, z])
plt.scatter(df["bill_length_mm"], df['bill_depth_mm'],
            #c = my_color
            c = my_color
            )
plt.xlabel('bill length (mm)')
plt.ylabel('bill depth (mm)')
plt.show()

df['species'] = df['species'].astype('category')
df.info()

df['species'].cat.categories
df['species'].cat.codes

plt.scatter(df["bill_length_mm"], df['bill_depth_mm'],
            #c = my_color
            c = df['species'].cat.codes
            )
plt.xlabel('bill length (mm)')
plt.ylabel('bill depth (mm)')
plt.show()

# 범주형 데이터 시각화
# Matplotlib은 범주형 변수를 자동으로 처리할 수 있습니다.
names = ['A', 'B', 'C']
values = [1, 10, 100]
plt.figure(figsize=(9, 3))
plt.subplot(231) 
plt.bar(names, values)  # 막대 그래프
plt.subplot(235)
plt.scatter(names, values)  # 산점도
plt.subplot(233)
plt.plot(names, values)  # 선 그래프
plt.suptitle('Categorical Plotting')
plt.show()


# 텍스트 추가 및 주석 처리
# 그래프에 중요한 내용을 강조할 때 사용됩니다.
#한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'   # 또는 'NanumGothic' 등
plt.rcParams['axes.unicode_minus'] = False       # 마이너스 기호 깨짐 방지


plt.plot([1, 2, 3, 4], [10, 20, 30, 40])
plt.text(2, 18, # 특정 위치에 텍스트 추가
        '중요한 Point', 
        fontsize=12, 
        color='red')
plt.show()

# 제목, X축, Y축, 범례 지정
# 그래프를 더욱 가독성 있게 만들기 위해 제목, 축 레이블, 범례 추가합니다.

plt.plot([1, 2, 3, 4], 
         [1, 4, 9, 16], 
        label="y = x^2")  
plt.title("Example Plot") # 재목 설정 
plt.xlabel("X Axis") # 축 라벨 
plt.ylabel("Y Axis") # 축 라벨  
plt.legend(loc="upper left") # 범례 표시
plt.show()

import pandas as pd

df = pd.read_csv('./data/penguins.csv')
df.info()

mean_bill_length = df.groupby("species")["bill_length_mm"].mean()
plt.bar(mean_bill_length.index, mean_bill_length.values)
plt.xlabel('펭귄의 종')
plt.ylabel('길이')
plt.title('종별 평균 부리 길이')
plt.show()

# 펭귄데이터 불러오자
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./data/penguins.csv')
df.info()

# 한글 폰트 설정 (예: 맑은 고딕, 필요 시 다른 폰트로 교체)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 섬별 평균 몸무게 계산
island_mean_mass = df.groupby('island')['body_mass_g'].mean().reset_index()

# 최대, 최소 인덱스 구하기
max_index = island_mean_mass['body_mass_g'].idxmax()
min_index = island_mean_mass['body_mass_g'].idxmin()

# 색상 설정
colors = []
for i in range(len(island_mean_mass)):
    if i == max_index:
        colors.append('red')       # 가장 무거운 섬
    elif i == min_index:
        colors.append('blue')      # 가장 가벼운 섬
    else:
        colors.append('gray')      # 나머지

# 시각화
plt.figure(figsize=(8, 6))
bars = plt.bar(island_mean_mass['island'], island_mean_mass['body_mass_g'], color=colors)

# 제목 및 축 라벨
plt.title('섬별 평균 몸무게')
plt.xlabel('섬')
plt.ylabel('평균 몸무게 (g)')

# 막대 위에 몸무게 표시
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 10,
             f'{height:.0f}g', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# 펭귄 종별 부리길이(x), 깊이(y) 산점도
# 한글 제목, x, y 축 제목
# 아델리 - 빨강
# 친스트랩 - 회색
# 겐투 - 회색
# 범례: 오른쪽 하단 위치
# 아델리 평균 중심점 표시
# 점찍고 텍스트로 아래와같이 출력
# ( 평균 부리길이: xx.xx mm,
# 평균 부리깊이: xx.xx mm)

import pandas as pd
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 결측값 제거
valid_df = df.dropna(subset=['bill_length_mm', 'bill_depth_mm', 'species'])

# 색상 매핑
color_map = {
    'Adelie': 'red',
    'Chinstrap': 'gray',
    'Gentoo': 'gray'
}

# 산점도 그리기
plt.figure(figsize=(9, 7))
for species, group in valid_df.groupby('species'):
    plt.scatter(group['bill_length_mm'], group['bill_depth_mm'],
                label=species, color=color_map.get(species, 'black'), alpha=0.7)

# 아델리 평균 계산
adelie_df = valid_df[valid_df['species'] == 'Adelie']
adelie_mean_x = adelie_df['bill_length_mm'].mean()
adelie_mean_y = adelie_df['bill_depth_mm'].mean()

# 평균점 표시 (검정색)
plt.scatter(adelie_mean_x, adelie_mean_y, color='black', s=100, zorder=5, marker='x')

# 평균 텍스트 (그래프 아래쪽 외부)
text_x = adelie_mean_x  # x 위치는 평균과 같게
text_y = plt.ylim()[0] - 0.5  # 현재 y축 최소값보다 조금 아래

annotation_text = (
    f'평균 부리길이: {adelie_mean_x:.2f} mm\n'
    f'평균 부리깊이: {adelie_mean_y:.2f} mm'
)

# 화살표 + 텍스트 추가
plt.annotate(
    annotation_text,
    xy=(adelie_mean_x, adelie_mean_y),        # 화살표 시작점 (평균점)
    xytext=(text_x-3, text_y+3),                  # 텍스트 위치
    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
    ha='center', va='top', fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.8)
)

# 제목 및 라벨
plt.title('펭귄 종별 부리 길이와 깊이 산점도')
plt.xlabel('부리 길이 (mm)')
plt.ylabel('부리 깊이 (mm)')

# 범례: 오른쪽 하단
plt.legend(loc='lower right', title='종')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.show()


#연속적인 값을 가지는 변수들을 쪼개서 범주형 변수로 변환시키는 테크닉













import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches # 범례 생성을 위한 패치 임포트

# 데이터 로드 (파일 경로를 확인해주세요)
df = pd.read_csv('./data/penguins.csv')

print("원본 데이터프레임 정보:")
df.info()
print("\n")

# 'sex' 칼럼의 값들을 대문자로 통일하고, 결측치 및 유효하지 않은 값 제거
df['sex'] = df['sex'].astype(str).str.upper().str.strip() # 대문자화 및 공백 제거
valid_sex = ['MALE', 'FEMALE']
df_cleaned = df[df['sex'].isin(valid_sex)].dropna(subset=['sex'])

print("결측치 제거 및 유효 성별 필터링 후 데이터프레임 정보:")
df_cleaned.info()
print("\n")

# 펭귄 종별 성별 카운트
species_sex_counts = df_cleaned.groupby(['species', 'sex']).size().unstack(fill_value=0)

print("종별 성별 카운트:")
print(species_sex_counts)
print("\n")

# 펭귄 종별 파이차트 생성 및 단일 범례 추가
plt.style.use('seaborn-v0_8-darkgrid') # 시각화 스타일 설정
plt.rcParams['font.family'] = 'Malgun Gothic' # 한글 폰트 설정 (Windows 기준)
# Mac 사용자는 'AppleGothic' 또는 'Apple SD Gothic Neo'로 변경
plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지

num_species = len(species_sex_counts.index)
# 서브플롯 개수에 따라 그림 크기 조정. 범례를 위해 아래쪽에 여유를 줍니다.
fig, axes = plt.subplots(1, num_species, figsize=(6 * num_species, 6), constrained_layout=True)

# 서브플롯이 하나일 경우 axes가 1차원 배열이 아닐 수 있으므로 리스트로 변환
if num_species == 1:
    axes = [axes]

# 성별에 따른 일관된 색상 정의
# 파이 차트의 섹션 순서는 데이터가 들어오는 순서(일반적으로 알파벳 순서)에 따라 달라질 수 있으므로,
# 'FEMALE'과 'MALE'에 각각 특정 색상을 지정하여 일관성을 유지합니다.
sex_colors_map = {'FEMALE': 'lightcoral', 'MALE': 'skyblue'}
# 범례에 사용될 순서대로 라벨과 색상 리스트 생성
legend_labels = ['FEMALE', 'MALE']
legend_colors = [sex_colors_map['FEMALE'], sex_colors_map['MALE']]

for i, species in enumerate(species_sex_counts.index):
    # 해당 종의 성별 데이터 가져오기
    sex_data = species_sex_counts.loc[species]
    
    # 성별 데이터를 'FEMALE', 'MALE' 순서로 정렬하여 색상 매핑의 일관성 유지
    ordered_sex_data = pd.Series([sex_data.get('FEMALE', 0), sex_data.get('MALE', 0)], index=['FEMALE', 'MALE'])
    
    # 총합이 0보다 큰 경우에만 차트 그리기
    if ordered_sex_data.sum() > 0:
        axes[i].pie(ordered_sex_data,
                    labels=None, # 직접 라벨을 표시하지 않고 범례로 대체
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=[sex_colors_map[s] for s in ordered_sex_data.index], # 색상 맵 적용
                    pctdistance=0.85) # 퍼센트 텍스트 위치 조정
        
        # 제목 설정
        axes[i].set_title(f'{species} 종 성비', fontsize=16)
        
        # 파이차트가 원형으로 보이도록 설정
        axes[i].axis('equal') 
    else: # 해당 종의 데이터가 없거나 성별 정보가 부족한 경우
        axes[i].text(0.5, 0.5, "데이터 부족",
                     horizontalalignment='center', verticalalignment='center',
                     transform=axes[i].transAxes, fontsize=14, color='gray')
        axes[i].set_title(f'{species} 종 성비', fontsize=16)

# 단일 범례 생성
legend_handles = [mpatches.Patch(color=color, label=label) 
                  for label, color in zip(legend_labels, legend_colors)]

# Figure의 오른쪽 하단에 범례 배치
# bbox_to_anchor=(1, 0)은 Figure의 오른쪽 하단 모서리를 의미하며,
# loc='lower right'는 범례 상자의 'lower right' 부분이 이 앵커에 정렬됨을 의미합니다.
fig.legend(handles=legend_handles, 
           labels=legend_labels, 
           title="성별", 
           loc="lower right", 
           fontsize=12,
           bbox_to_anchor=(0.98, 0.02), # Figure 우측 하단에서 약간 안쪽으로 조정
           frameon=True, # 범례 테두리 표시
           borderaxespad=0. # bbox_to_anchor와의 간격 조정
          )

plt.show()




import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler # 데이터 정규화를 위해 사용

# 데이터 로드 (파일 경로를 확인해주세요)
try:
    df = pd.read_csv('./data/penguins.csv')
except FileNotFoundError:
    print("오류: 'penguins.csv' 파일을 찾을 수 없습니다. 예시 데이터를 사용하여 진행합니다.")
    # 파일이 없는 경우를 대비한 예시 데이터 (실제 penguins.csv와 동일한 변수명 사용)
    data = {
        'species': ['Adelie', 'Adelie', 'Adelie', 'Chinstrap', 'Chinstrap', 'Chinstrap', 'Gentoo', 'Gentoo', 'Gentoo'],
        'island': ['Torgersen', 'Torgersen', 'Biscoe', 'Dream', 'Dream', 'Dream', 'Biscoe', 'Biscoe', 'Biscoe'],
        'bill_length_mm': [39.1, 39.5, 40.3, 46.5, 50.3, 49.6, 49.1, 50.1, 47.7],
        'bill_depth_mm': [18.7, 17.4, 18.0, 17.9, 20.0, 18.2, 14.8, 15.2, 16.6],
        'flipper_length_mm': [181.0, 186.0, 195.0, 195.0, 202.0, 201.0, 212.0, 220.0, 212.0],
        'body_mass_g': [3750.0, 3800.0, 3250.0, 4400.0, 5550.0, 5200.0, 5500.0, 6300.0, 5000.0],
        'sex': ['MALE', 'FEMALE', 'FEMALE', 'MALE', 'FEMALE', 'MALE', 'MALE', 'FEMALE', 'FEMALE']
    }
    df = pd.DataFrame(data)

print("원본 데이터프레임 정보:")
df.info()
print("\n")

# 레이더 차트에 사용할 변수들 정의
features = ['bill_length_mm', 'flipper_length_mm', 'bill_depth_mm', 'body_mass_g']

# 결측치 제거 (레이더 차트 생성 시 결측치는 오류를 유발할 수 있음)
df_cleaned = df.dropna(subset=features + ['species'])

print("결측치 제거 후 데이터프레임 정보:")
df_cleaned.info()
print("\n")

# 각 종(species)별 평균값 계산
df_species_avg = df_cleaned.groupby('species')[features].mean().reset_index()

print("종별 평균값:")
print(df_species_avg)
print("\n")

# 데이터 정규화 (Min-Max Scaling)
# 각 특성별 스케일을 0에서 1 사이로 맞춤
scaler = MinMaxScaler()
# 평균값만 가진 DataFrame에서 features 컬럼에 대해서만 정규화 적용
df_normalized = pd.DataFrame(scaler.fit_transform(df_species_avg[features]), columns=features)
df_normalized['species'] = df_species_avg['species'] # 정규화된 데이터에 species 컬럼 다시 추가

print("정규화된 종별 평균값 (0~1 스케일):")
print(df_normalized)
print("\n")

# 레이더 차트 그리기
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'Malgun Gothic' # 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 깨짐 방지

# 차트 축 설정 (변수 개수에 따라 각도 조정)
num_vars = len(features)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() # 2*pi (360도)를 변수 개수만큼 나눔
angles += angles[:1] # 첫 번째 지점으로 다시 돌아와 닫힌 그래프를 만듦

# Figure와 Axes 생성 (극좌표계를 사용)
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# 각 종별로 레이더 차트 그리기
species_colors = {'Adelie': 'blue', 'Chinstrap': 'green', 'Gentoo': 'red'} # 종별 색상 설정
species_markers = {'Adelie': 'o', 'Chinstrap': 's', 'Gentoo': '^'} # 종별 마커 설정

for idx, row in df_normalized.iterrows():
    values = row[features].tolist()
    values += values[:1] # 첫 번째 값 다시 추가하여 그래프 닫기

    species_name = row['species']
    ax.plot(angles, values, color=species_colors[species_name], 
            marker=species_markers[species_name], linewidth=2, 
            label=f'{species_name} 펭귄')
    ax.fill(angles, values, color=species_colors[species_name], alpha=0.25) # 면 채우기

# 축 라벨 설정
ax.set_theta_offset(np.pi / 2) # 시작점을 위로 (0도를 위로)
ax.set_theta_direction(-1) # 시계 방향으로
ax.set_xticks(angles[:-1]) # 마지막 반복되는 각도 제외
ax.set_xticklabels(['부리 길이 (mm)', '날개 길이 (mm)', '부리 깊이 (mm)', '몸무게 (g)'], fontsize=12)

# Y축 (반경) 설정
# 정규화된 값이 0에서 1 사이이므로, 축 눈금도 0.25 단위 등으로 설정
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], color="grey", size=10)
ax.set_ylim(0, 1) # y축 범위 0에서 1로 설정

# 제목 및 범례 추가
ax.set_title('펭귄 종별 신체 특성 비교 (정규화된 값)', y=1.1, fontsize=18)
ax.legend(loc='lower left', bbox_to_anchor=(0.9, 0.95), ncol=1, fontsize=12) # 범례 위치 조정

plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 데이터 로드 (파일 경로를 확인해주세요)
try:
    df = pd.read_csv('./data/penguins.csv')
except FileNotFoundError:
    print("오류: 'penguins.csv' 파일을 찾을 수 없습니다. 예시 데이터를 사용하여 진행합니다.")
    # 파일이 없는 경우를 대비한 예시 데이터 (실제 penguins.csv와 동일한 변수명 사용)
    data = {
        'species': ['Adelie', 'Adelie', 'Adelie', 'Chinstrap', 'Chinstrap', 'Chinstrap', 'Gentoo', 'Gentoo', 'Gentoo', 'Adelie'],
        'island': ['Torgersen', 'Torgersen', 'Biscoe', 'Dream', 'Dream', 'Dream', 'Biscoe', 'Biscoe', 'Biscoe', 'Biscoe'],
        'bill_length_mm': [39.1, 39.5, 40.3, 46.5, 50.3, 49.6, 49.1, 50.1, 47.7, 39.0],
        'bill_depth_mm': [18.7, 17.4, 18.0, 17.9, 20.0, 18.2, 14.8, 15.2, 16.6, 17.0],
        'flipper_length_mm': [181.0, 186.0, 195.0, 195.0, 202.0, 201.0, 212.0, 220.0, 212.0, 189.0],
        'body_mass_g': [3750.0, 3800.0, 3250.0, 4400.0, 5550.0, 5200.0, 5500.0, 6300.0, 5000.0, 3400.0],
        'sex': ['MALE', 'FEMALE', 'FEMALE', 'MALE', 'FEMALE', 'MALE', 'MALE', 'FEMALE', 'FEMALE', 'MALE']
    }
    df = pd.DataFrame(data)

print("원본 데이터프레임 정보:")
df.info()
print("\n")

# 레이더 차트에 사용할 변수들 정의
features = ['bill_length_mm', 'flipper_length_mm', 'bill_depth_mm', 'body_mass_g']

# 결측치 제거
# 레이더 차트 생성 시 결측치는 오류를 유발할 수 있으므로, 해당 변수들과 species 칼럼에 결측치가 있는 행 제거
df_cleaned = df.dropna(subset=features + ['species'])

print("결측치 제거 후 데이터프레임 정보:")
df_cleaned.info()
print("\n")

# 각 종(species)별 평균값 계산
df_species_avg = df_cleaned.groupby('species')[features].mean().reset_index()

print("종별 평균값:")
print(df_species_avg)
print("\n")

# 점수 부여 로직 (순위 기반)
df_scores = pd.DataFrame({'species': df_species_avg['species']})
scoring_map = {1: 4, 2: 3, 3: 2, 4: 1} # 1등:4점, 2등:3점, 3등:2점, 그외:1점

for feature in features:
    # 해당 특성의 평균값을 기준으로 내림차순 정렬하여 순위 부여
    # rank(method='min')은 동점일 경우 같은 순위 부여 후 다음 순위를 건너뜀
    # (예: 1등, 1등, 3등).
    # 여기서는 'method='dense''를 사용하여 동점이어도 순위를 연속적으로 부여 (예: 1등, 1등, 2등)
    # 펭귄 3종이라 동점시 4점, 4점, 2점 이렇게 될 수 있음
    # 또는 'first'를 사용하면 값이 같을 경우 데이터프레임 순서에 따라 순위를 매깁니다.
    # 단순 1등, 2등, 3등을 명확히 하려면 'first'가 더 직관적일 수 있습니다.
    
    # 여기서는 평균값이 높을수록 높은 점수를 주는 방식이므로 내림차순으로 순위를 매깁니다.
    ranked_series = df_species_avg[feature].rank(method='dense', ascending=False)
    
    # 순위에 따라 점수 부여
    df_scores[feature] = ranked_series.map(scoring_map).fillna(1).astype(int) # 맵에 없는 순위는 1점으로 처리

print("종별 점수 (순위 기반):")
print(df_scores)
print("\n")

# 레이더 차트 그리기
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'Malgun Gothic' # 한글 폰트 설정 (Windows 기준, Mac은 'AppleGothic')
plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 깨짐 방지

# 차트 축 설정 (변수 개수에 따라 각도 조정)
num_vars = len(features)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() # 2*pi (360도)를 변수 개수만큼 나눔
angles += angles[:1] # 첫 번째 지점으로 다시 돌아와 닫힌 그래프를 만듦

# Figure와 Axes 생성 (극좌표계를 사용)
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# 각 종별로 레이더 차트 그리기
# 색상과 마커는 시각적 구분을 위해 설정
species_colors = {'Adelie': 'blue', 'Chinstrap': 'green', 'Gentoo': 'red'}
species_markers = {'Adelie': 'o', 'Chinstrap': 's', 'Gentoo': '^'}

# 특성 라벨 (사용자에게 친숙한 이름)
feature_labels = ['부리 길이 (점수)', '날개 길이 (점수)', '부리 깊이 (점수)', '몸무게 (점수)']

for idx, row in df_scores.iterrows():
    # scores 리스트 생성 및 첫 번째 값 다시 추가하여 그래프 닫기
    scores = row[features].tolist()
    scores += scores[:1] 

    species_name = row['species']
    ax.plot(angles, scores, color=species_colors[species_name], 
            marker=species_markers[species_name], linewidth=2, 
            label=f'{species_name} 펭귄')
    ax.fill(angles, scores, color=species_colors[species_name], alpha=0.25) # 면 채우기

# 축 라벨 설정 (각 축에 해당하는 변수 이름을 점수임을 명시하여 표시)
ax.set_theta_offset(np.pi / 2) # 시작점을 위로 (0도를 위로)
ax.set_theta_direction(-1) # 시계 방향으로
ax.set_xticks(angles[:-1]) # 마지막 반복되는 각도 제외
ax.set_xticklabels(feature_labels, fontsize=12)

# Y축 (반경) 설정
# 점수가 1점에서 4점 사이이므로 Y축 눈금을 명확하게 설정
ax.set_yticks([1, 2, 3, 4])
ax.set_yticklabels(['1점', '2점', '3점', '4점'], color="grey", size=10)
ax.set_ylim(1, 4) # Y축 범위 1에서 4로 설정

# 제목 및 범례 추가
ax.set_title('펭귄 종별 신체 특성 순위 점수 비교', y=1.1, fontsize=18)
ax.legend(loc='lower left', bbox_to_anchor=(0.9, 0.95), ncol=1, fontsize=12) # 범례 위치 조정

plt.show()

df.info()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv('./data/penguins.csv')

# 사용할 변수
metrics = ['bill_length_mm', 'bill_depth_mm', 'body_mass_g', 'flipper_length_mm']

# 성별별 평균 계산 (NaN 자동 제외)
mean_df = df.groupby('sex')[metrics].mean().dropna()

# 등수 기반 점수 부여: 평균값이 높은 성별에게 높은 점수
score_df = pd.DataFrame(index=mean_df.index)

for col in metrics:
    # 내림차순 정렬 → 큰 값일수록 높은 등수
    ranked = mean_df[col].rank(ascending=False, method='min')
    # 등수 → 점수 매핑 (1등: 3점, 2등: 2점, 3등: 1점)
    score_df[col] = ranked.map({1: 3, 2: 2, 3: 1})

# 레이더 차트 준비
labels = metrics
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# 차트 그리기
plt.figure(figsize=(8, 6))
ax = plt.subplot(111, polar=True)

# 각 성별에 대해 레이더 플롯
for sex in score_df.index:
    values = score_df.loc[sex].tolist()
    values += values[:1]  # 닫기용
    ax.plot(angles, values, label=sex)
    ax.fill(angles, values, alpha=0.25)

# 축 설정
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12)
ax.set_yticks([1, 2, 3])
ax.set_yticklabels(['1점', '2점', '3점'], fontsize=10)
plt.title('펭귄 성별 특성별 평균 등수 기반 점수 (1등=3점)', size=14)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
plt.grid(True)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 (윈도우 기준 예시)
plt.rcParams['font.family'] = 'Malgun Gothic'

# 필요한 열만 선택하고 결측치 제거
cols = ['species', 'bill_length_mm', 'bill_depth_mm', 'body_mass_g', 'flipper_length_mm']
df_clean = df[cols].dropna()

# 종별 평균 계산
grouped = df_clean.groupby('species').mean()

# 각 변수별 순위 계산 (높을수록 좋은 점수)
ranked = grouped.rank(ascending=False)
n_species = grouped.shape[0]
scores = ranked.apply(lambda x: (n_species + 1) - x)

# 레이더 차트 변수
labels = scores.columns.tolist()
num_vars = len(labels)

# 각 각도 계산
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 첫번째 값으로 닫기

# 점수 값도 레이더에 맞게 닫기
def create_radar_values(row):
    values = row.tolist()
    values += values[:1]
    return values

# 플롯 설정
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# 종별로 그래프 그리기
for species, row in scores.iterrows():
    values = create_radar_values(row)
    ax.plot(angles, values, label=species)
    ax.fill(angles, values, alpha=0.2)

# 축 라벨 설정
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

# y축 범위 설정
ax.set_ylim(0, n_species)

# 제목 및 범례
ax.set_title('펭귄 종별 특성 비교 (등수 기반 점수)', size=14)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.show()
