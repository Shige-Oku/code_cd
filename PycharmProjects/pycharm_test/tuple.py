#coding: UTF-8
a = [2012, 2013, 2014]
b = (2012, 2013, 2014)
print(a)
print(b)

print(a[1])
print(b[1])

# list は変更可
a[1] = 2016
print(a)

# tupleは変更できない
# b[1] = 2016
# print(b)

a.append(2015)
print(a)

# tupleは変更不可なのでappendは定義されていない
# b.append(2015)
# print(b)
