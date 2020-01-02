a = 7
b = 3

def func1(c, d):
    e = c + d
    e += c
    e *= d
    return e

f = func1(a, b)
print(f)