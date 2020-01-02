# coding: UTF-8

class Dog:
    name = ""
    def bark(self):
        m = self.name + "Bow wow!"
        print(m)

class Shibainu(Dog):
    age = 0
    def sayAge(self):
        m = "I'm " + str(self.age) + " years old"
        print(m)

hachi = Shibainu()
hachi.name   = "Hachi"
hachi.age = 5
hachi.bark()
hachi.sayAge()
