#coding: UTF-8

a = range(0, 10)
print(list(a))

b = []
for i in a:
    if i % 2 == 0:
        b.append(i)

print(b)
