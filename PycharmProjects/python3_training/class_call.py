# coding: UTF-8

class Dog:
    def __init__(self, na):
        self.name = na

    def bark(self):
        m = self.name + ": Bow-wow!"
        print(m)

    def __call__(self, ag, wg):
        m = "Name: " + self.name + " Age: " + str(ag) + " Weight: " + str(wg)
        print(m)

pochi = Dog("Pochi")
pochi.bark()
pochi(5, 20)
