# coding: UTF-8

class Dog:
    # name = ""
    def __init__(self, nm):
        self.name = nm

    def bark(self):
        m = self.name + ": Bow wow!"
        print(m)

pochi = Dog("Pochi")
pochi.bark()

class Shibainu(Dog):
    def __init__(self, nm, ag):
        super().__init__(nm)
        self.age = ag

    def sayAboutMe(self):
        m = self.name + ": I'm " + str(self.age)+ " years old!"
        print(m)

hachi = Shibainu("Hachi", 5)
hachi.sayAboutMe()

class Mameshiba(Shibainu):
    def __init__(self, nm, ag, wg):
        super().__init__(nm, ag)
        self.weight = wg

    def myProfile(self):
        m = "My profile = " + str(self.weight) + " kg"
        print(m)

machi = Mameshiba("Machi", 3, 15)
machi.myProfile()

