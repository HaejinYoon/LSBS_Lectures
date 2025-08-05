x = 15
type(x)

# 리스트 생성 예제
fruits = ['apple', 'banana', 'cherry']
numbers = [1, 2, 3, 4, 5]
mixed_list = [1, "Hello", [1, 2, 3]]
print("Fruits:", fruits)

fruits[0]
fruits[1:]

numbers + numbers
numbers * 3

numbers + fruits + mixed_list

mixed_list[2][1]
len(numbers)
len(mixed_list)

numbers.append(200)
numbers.append([1,4])
del numbers[-1]
numbers[0] = 20
numbers.sort()


numbers

# 튜플 생성 예제
a = (10, 20, 30) # a = 10, 20, 30 과 동일
b = (42,)
a[1:]
a + b

d = (2,)
type(d)

a[2] = 3

tup_e = (1, 3, "a", [1, 4], (2, 4))
tup_e


#1
kor = 80
eng = 75
mat = 55

(kor + eng + mat)/3

#2
13%2 == 0

#3
pin = "881120-1068234"

yyyymmdd = "19"+pin[0:6]
num = pin[7:]
yyyymmdd
num

#4
pin[7]

#5
a = "a:b:c:d"
b= a.replace(':', "#")
b

#6
a = [1, 3, 5,  4, 2]
a.sort()
a.reverse()
a

#7
a = ['Life', 'is', 'too', 'short']
result = " ".join(a)
result

#8
a = (1, 2, 3)
a = a + (4,)
print(a)

# 딕셔너리 생성 예제
person = {
    'name': 'John',
    'age': 30,
    'city': 'New York'
}
print("Person:", person)
person[0] # 인덱스 지원 안됨
person.get('name')
person.keys()

a = {1 : 'hi'}
a[2] = 'b'
a['name'] = 'Harry'
a['name'] = 'Harry Potter'
del a[2]
a.get(1)
a[1]
list(a.keys())

list(a.values())

'name' in a

s1 = set([1, 2, 3, 2, 2, 3])
list(s1)

s1 = set([1, 2, 3, 4, 5, 6])
s2 = set([4, 5, 6, 7, 8, 9])

s1 & s2
s1 | s2

#Q9
a = dict()
a['name'] = 'python'
a[('a',)] = 'python'
a[[1]] = 'python'
a[250] = 'python'

#Q10
a = {'A' : 90, 'B' : 80, 'C':70}
result = a.pop('B')
a
result

#Q11
a=[1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5]
aSet = set(a)
b = list(aSet)
