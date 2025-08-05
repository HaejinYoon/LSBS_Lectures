#변수 개념 이해하기 연습문제
#1
city_name = "Yongin"
age = 34
is_student = False

#2
#a. user_name

#3
x = 12
y= 4

print((x - y) * y / x)
print(x ** y // x)
print((x % y) + 2)

#4
age = 34
is_raining = False
is_warm = True

print("18세 이상 여부 ",age > 18)

print("비가 오지 않고 따뜻한 경우 ", is_raining and is_warm)
print("비가 오거나 따뜻한 경우 ", is_raining or is_warm)
print("비가 오지 않는 경우 ",is_raining)

#5
price = 12000
quantity = 3
total_price = price * quantity
print(total_price >= 10000 and total_price <= 50000)

#데이터 타입 이해하기 연습문제
#1
num = 100
pi = 3.14
name = "111"
fruits = ["apple", "banana", "cherry"]
data = (10, 20, 30)
person = {"name": "Tom", "age": 25}
flags = {True, False}

count = 0
result = [
    type(num) == int,
    type(pi) == float,
    type(name) == int,
    type(fruits) == list,
    type(data) == tuple,
    type(person) == dict,
    type(flags) == set,
]
result
print('True count >>>> ',result.count(True))

#2
nums = [5, 10, 10, 15 ]
nums = tuple(nums)
nums = set(nums)
len(nums)

#3
profile = {
    "name": "Jane",
    "age": 27,
    "city": "Busan"
}

profile["age"] = 28
del profile["city"]
profile

#4
set_x = {1, 2, 3, 4}
set_y = {3, 4, 5, 6}

set_x | set_y
set_x & set_y
set_x - set_y


#5
sentence = "Python Is FUN"
tmp = sentence.split()
tmp
tmp[0] = tmp[0].upper()
tmp[1] = tmp[1].lower()
tmp[2] = tmp[2].lower()
sentence = " ".join(tmp)

up = sentence[:7].upper()
lower = sentence[7:].lower()
sentence = up + lower
sentence

#교재 119페이지 12번 연습문제
a = b = [1, 2, 3]
a[1] = 4
print(b)

a = [1, 2, 3]
b = [1, 2, 3]
a[1] = 4
print(a)
print(b)
