import pandas as pd
data = {
    '가전제품': ['냉장고', '세탁기', '전자레인지', '에어컨', '청소기'],
    '브랜드': ['LG', 'Samsung', 'Panasonic', 'Daikin', 'Dyson']
}
df = pd.DataFrame(data)
df

df.info()

df['제품명 길이'] = df["가전제품"].str.len()
df['브랜드명 길이'] = df["브랜드"].str.len()
df

df['브랜드'] = df['브랜드'].str.lower()

df['브랜드'].str.contains('i')
df['브랜드'].str.contains('s')
df['브랜드'].str.contains('g')

df['가전제품'].str.replace("에어컨", "선풍기")
df['가전제품'].str.replace("기", "")
df

df['브랜드'].str.split("a", expand=True)


df['제품_브랜드'] = df['가전제품'].str.cat(df['브랜드'], sep=', ')
df

df['가전제품'] = df['가전제품'].str.replace('전자레인지', ' 전자 레인지  ')
df['가전제품'] .str.strip()
df['가전제품'] .str.replace(" ", "")


data = {
    '주소': ['서울특별시 강남구 테헤란로 123', '부산광역시 해운대구 센텀중앙로 45', '대구광역시 수성구 동대구로 77-9@@##', '인천광역시 남동구 예술로 501&amp;&amp;, 아트센터', '광주광역시 북구 용봉로 123']
}
df = pd.DataFrame(data)
print(df.head(2))

df['도시'] = df['주소'].str.extract(r'([가-힣]+광역시|[가-힣]+특별시)', expand=False)
df

special_chars = df['주소'].str.extractall(r'([^a-zA-Z0-9가-힣\s])')
print(special_chars)

df['주소_특수문자제거'] = df['주소'].str.replace(r'[^a-zA-Z0-9가-힣\s]', '', regex=True)
print(df.head(2))
#amp 제거
df['주소_특수문자제거'] = df['주소_특수문자제거'].str.replace("amp", '')

df





df = pd.DataFrame({
    'text': [
        'apple',        # [aeiou], (a..e), ^a
        'banana',       # [aeiou], (ana), ^b
        'Hello world',  # ^Hello, world$
        'abc',          # (abc), a.c
        'a1c',          # a.c
        'xyz!',         # [^aeiou], [^0-9]
        '123',          # [^a-z], [0-9]
        'the end',      # d$, e.
        'space bar',    # [aeiou], . (space)
        'hi!',           # [^0-9], [aeiou]
        'blue',
        'lue'
    ]
})
df['text'].str.extract(r'([aeiou])')
df['text'].str.extractall(r'([aeiou])')
df['text'].str.extract(r'([^0-9])')
df['text'].str.extractall(r'([^0-9])')

df['text'].str.extract(r'(ba)')
df['text'].str.extract(r'(a.c)')
df['text'].str.extract(r'(^Hello)')
df['text'].str.extract(r'(b?lue)')

#[연습문제] 정규표현식 연습하기
df = pd.read_csv("./data/regex_practice_data.csv")
#1
df["이메일"] = df["전체_문자열"].str.extract(r'([^a-zA-Z0-9])*?@')
df["이메일"] = df["전체_문자열"].str.extract(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})')
r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
df


#2
df["전체_문자열"].str.extract(r'(+\-[0-9]+\-[0-9]*)')
df["전체_문자열"].str.extract(r'((010-[0-9]+-))')

df["전화번호"] = df["전체_문자열"].str.extract(r'([0-9]+\-[0-9]+\-[0-9]*)')
df

#3
