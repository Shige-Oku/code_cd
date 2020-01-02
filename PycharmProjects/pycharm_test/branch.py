#coding: UTF-8
a = 5

if a == 5 :
    print("a = 5")
else:
    print("a not = 5")

b = 4
if b < 3 :
    print("b < 3")
elif b < 5 : # else if
    print("3 < b < 5")
else:
    print("b > 5")

time = 15

if time > 5 and time < 12 :
    print("Good morning!")
elif time >= 12 and time < 18 :
    print("Good afternoon!")
else:
    print("Good evening!")
