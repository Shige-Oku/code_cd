# coding: UTF-8

class Calculation():
    Value = 0
    def squere(self):
        s = self.Value * self.Value
        return s

# a = Calculation()
# b = Calculation()
# c = Calculation()
#
# d = [a, b, c]

d = [Calculation(), Calculation(), Calculation()]

d[0].Value = 3
d[1].Value = 5
d[2].Value = 7

print(d[0].squere())
print(d[1].squere())
print(d[2].squere())

for e in d:
    print(e.squere())

