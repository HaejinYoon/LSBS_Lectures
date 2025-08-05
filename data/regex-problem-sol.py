import pandas as pd
#### 정규표현식 연습 ####
df=pd.read_csv("./regex_practice_data.csv")
df=df[['전체_문자열']]

# 이메일 잡아오기
df['전체_문자열'].str.extract(r'([\w\.]+@[\w\.]+)')

# 핸드폰 번호 잡아오기 + 핸드폰 번호 입력한 사람들 정보
df['전체_문자열'].str.extract(r'(010-[0-9\-]+)').dropna()

# 일반 번호 잡아오기
phone_num=df['전체_문자열'].str.extract(r'(\d+-[0-9\-]+)')
phone_num.iloc[:,0]
~phone_num.iloc[:,0].str.startswith("01")
phone_num.loc[~phone_num.iloc[:,0].str.startswith("01"),:]

# 주소에서 '구' 단위만 추출하기
df['전체_문자열'].str.extract(r'(\b\w+구\b)')
df['전체_문자열'].str.extract(r'([가-힣]+구)')

# 날짜(YYYY-MM-DD) 형식 찾기
df['전체_문자열'].str.extract(r'(\d{4}-\d{2}-\d{2})')

# 날짜 형식 가져오기
df['전체_문자열'].str.extract(r'(\d{4}\W\d{2}\W\d{2})')
df['전체_문자열'].str.extract(r'(\d{4}[-/.]\d{2}[-/.]\d{2})')

# 가격 정보(₩ 포함) 찾기
df['전체_문자열'].str.extract(r'(₩[\d,]+)')

# 가격에서 숫자만 추출하기 (₩ 제거)
df['전체_문자열'].str.extract(r'₩([\d,]+)')
# df['전체_문자열'].str.extract(r'₩(\d+\,?[\d,]+)')

# 이메일의 도메인 추출하기
# @
df['전체_문자열'].str.extract(r'@([\w.]+)')

# 데이터에서 한글 이름만 추출하세요.
df['전체_문자열'].str.extract(r'([가-힣]+)')