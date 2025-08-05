def add(a, b):
    result = a + b
    return result

add(3, 4)

# 매개변수: parameters
# 인수: argument

def say():
    return "hi"

say()

#p.121
#if

money = 5000
card = True
if((money > 4500) or (card == True)):
    print("택시를 타세요")
else:
    print("걸어가세요")

#시험 점수가 60점 이상이면 합격 그렇지 않으면 불합격을 출력하는 if문 작성
score = 50
if(score >= 60):
    print("합격")
else:
    print("불합격")

# x의 수가 홀수이면서 7의 배수이면 "7의 배수이면서 홀수"를 출력, 그렇지 않으면, "조건 불만족" 출럭
x = 14

if((x %2 ==1) and (x%7 ==0  )):
    print("7의 배수이면서 홀수")
else:
    print("조건 불만족")


# M이면 "남자입니다"
# F이면 "여자입니다"
# nan이면 "비어있습니다"

gender = "M"

if(gender == "M"):
    print("남자입니다")
elif(gender == "F"):
    print("여자입니다")
else:
    print("비어있습니다")

# 다음은 공원 입장료 정보입니다.
# 유아(7세 이하) : 무료
# 어린이: 3000원
# 성인: 7000원
# 노인(60세 이상): 5000원
# cal_price() 함수를 만들어보세요.
# 매개변수 age


age = 30
price = 0
def cal_price(age):
    if(age <= 7):
        print("유아 무료입니다.")
        price = 0
    elif(age >7 and age < 20):
        price =3000
    elif(age >= 20 and age <60):
        price =7000
    else:
        price =5000
    return price
cal_price(30)


# p.133 while 문
# 조건을 만족하는 동안(True) 코드를 반복실행
treeHit = 0
while treeHit < 10:
    treeHit +=1
    print(f"나무를 {treeHit}번 찍었습니다.")
    if treeHit == 10:
        print("나무가 넘아갑니다.")


treeHit = 0
while treeHit < 10:
    treeHit +=1
    print(f"나무를 {treeHit}번 찍었습니다.")
    if treeHit == 6:
        print("일 그만!")
        break

# break 와 continue
a = 0
while a < 10 :
    a += 1
    if(a%2 == 0):
        continue # while 루프 처음으로 넘어가
    print(a)

#for 루프
# for 변수 in 순서가 있는 객체:
#     반복할 내용1
#     반복할 내용2
#     .....

test_list = [ "one", "two", "three"]
for i in test_list:
    print(i)

a = [ (1, 2), (3, 4), (5, 6)]

for (fir, snd) in a:
    print(fir + snd)

#Q. 1에서 100까지 넘파이 벡터 만들기
import numpy as np

a= np.arange(1, 101)
for i in a:
    if (i % 7 == 0):
        continue
    print(i)
    
numbers = [x**2 for x in range(1, 6)]
numbers

a = [1, 2, 4, 3, 5]
#a의 각 원에 3을 곱한 값을 다시 리스트로 만들기
a * 3
for num in a:
    print(num*3)

[num*3 for num in a]

# 1 부터 10까지의 정수 중 각 수의 제곱값을 요소로 가지는 리스트
[x**2 for x in range(1, 11)]

# 1부터 20까지의 정수 중 짝수만 담은 리스트
[x for x in range(1, 21) if(x%2 == 0)]
for x in range(1, 21):
    if x%2 == 0:
        print(x)

#음수는 0으로 바꾸고 양수는 유지
nums = [-3, 5, -1, 0, 8]

for x in nums:
    if(x < 0):
        print(0)
    else:
        print(x)

modified = [ x if x >= 0 else 0 for x in nums]
modified = [ 0 if x < 0 else x for x in nums]

#리스트에서 a로 시작하는 문자열만 추출하기
words = ["apple", 'banana', 'cherry', 'avocado']
for i in words:
    if(i.startswith('a')):
        print(i)

[i for i in words if (i.startswith('a'))]
[i for i in words if (i.startswith('a'))]

"apple".startswith("a")
"banana".startswith("a")
"cherry".startswith("a")

#p.160
#def 함수(*par):
    #수행할 문장

def add_many(*nums):
    result=0
    for i in nums:
        result = result +i
    return result

add_many(3, 4, 2)

def cal_how(method, *nums):
    if method == "add":
        result = 0
        for i in nums:
            result += i
    elif method == "mul":
        result = 1
        for i in nums:
            result *= i
    else:
        print("해당 연산은 수행할 수 없습니다.")
        result = None
    return result

cal_how("add", 3, 4)
cal_how("mul", 3, 4, 7, 2)
cal_how("squared", 3, 4)

def add_and_mul(a=5, b=4):
    return a+b, a*b

add_and_mul(3, 4)
add_and_mul(b=3)
result = add_and_mul(3, 4)
result = add_and_mul(3)
result[0]
result[1]                                                                                                          














































