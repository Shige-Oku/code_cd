a = {"Taro":1985, "Hanako":1986}
print(a)

b = a["Taro"]
print(b)

a["Jiro"] = 1988
print(a)

a["Hanako"] = 1990
print(a)

c = {1:1991, 2:1992}
print(c)

d = c[2]
print(d)

print(iter(a))