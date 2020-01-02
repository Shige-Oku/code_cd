#coding: UTF-8
a = [2012, 2013, 2014]
print(a)

print(a[0])
print(a[1])
print(a[2])

b = 2012
c = [b, 2015, 20.1, "Hello", "Hi", 2015, 2000, 2015]
print(c[1:4])

print(c.count(2015))
print(c.count(2013))

# 指定したキーの位置を返却、キー、開始、終了
# 存在しないキーだと ValueError が発生する
print(c.index("Hello"))
# print(c.index("Hello", 5))
print(c.index("Hello",0 ,5))

print(c.reverse())

# print(c.sort(key=object))
