#coding: UTF-8
a = range(0, 10)
print(a)

b = []
for i in a:
    #  偶数だけリストに追加
    if i % 2 == 0:
        b.append(i)

print(b)