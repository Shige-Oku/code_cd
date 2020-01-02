#coding: UTF-8

a = 3
b = 7

def add1(c, d):
    e = c + d
    print(e)

add1(a, b)

def add2(c, d):
    e = c + d
    return e

f = add2(a, b)
print(f)
