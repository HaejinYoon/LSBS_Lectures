#파이선 함수와 반복문 알아보기 연습문제

#1
def add_numbers(a=1, b=2):
    return a + b

add_numbers(5, 7)

#2 
def check_sign(x):
    if x > 0 :
        print("Positive")
    elif x < 0:
        print("Negative")
    else:
        print(0)

check_sign(10)
check_sign(-5)
check_sign(0)

#3
def print_numbers():
    for i in range(1, 11):
        print(i)

print_numbers()

#4
def outer_function(x):
    def inner_function(y):
        print(y+2)
    return inner_function(x)

outer_function(5)

def find_even(start):
    while True:
        if(start %2 == 0):
            return start
        start +=1 

find_even(5)


#추가연습문제
#1 1부터 10까지 중 짝수의 제곱만 리스트에 담아서 출력
def func ():
    result = []
    for i in range(1, 11):
        if i % 2 ==0:
            result.append(i**2)
    return result
func()

#2 
customers = [
    {"name": "Alice", "age": 25},
    {"name": "Bob", "age": 34},
    {"name": "Charlie", "age": 29},
    {"name": "David", "age": 32},
    {"name": "Eve", "age": 22}
]
#2-1. 각 고객의 나이를 1년씩 증가시킨 새로운 리스트를 생성하세요.
newCus = []
for cus in customers:
    cus["age"] += 1
    newCus.append(cus)
newCus

#2-2. 30세 이상의 고객만 새로운 리스트로 필터링하세요.
newCus = []
for cus2 in customers:
    if cus2["age"] >= 30 :
        newCus.append(cus2)
newCus    

#2-3. 모든 고객의 나이를 더한 총 나이를 출력하세요.
cusAge = 0
for cus3 in customers:
    cusAge += cus3["age"]
cusAge

#2-4. 30세 미만의 고객 중 이름이 “A”로 시작하는 사람의 이름을 출력하세요.
startsA =[]
for cus4 in customers:
    if cus4["age"] <30:
        if cus4["name"].startswith('A'):
            startsA.append(cus4)
            print(cus4["name"])

#3
sales_data = {
    "January": 90,"February": 110,
    "March": 95,"April": 120,
    "May": 80,"June": 105,
    "July": 135,"August": 70,
    "September": 85,"October": 150,
    "November": 125,"December": 95
}

#3-1 판매량이 100 이상인 월을 필터링하여 출력하세요.
print('---------판매량이 100이상인 월---------')
for data in sales_data:
    if (sales_data.get(data) > 100):
        print(data)

#3-2. 연간 총 판매량과 월 평균 판매량을 계산하세요.
total = 0
avg = 0
for data in sales_data:
    total += sales_data.get(data)
avg = total/len(sales_data)

print('연간 총 판매량 : ', total)
print('월 평균 판매량 : ', avg)

        
#3-3. 판매량이 가장 높은 월의 이름과 판매량을 출력하세요.
max = 0
maxmonth = ''
for data in sales_data:
    if max < sales_data.get(data) :
        max = sales_data.get(data)
        maxmonth = data
print("판매량이 가장 높은 월 : ",maxmonth, max)

sales_data.items()
sales_data.values()


a = (10, 20, 30)
type(a)
a[:2]






























