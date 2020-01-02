class Dog():
    name = ""
    def bark(self):
        m = self.name + ": Bow-wow"
        print(m)
    def __init__(self, nm):
        self.name = nm

pochi = Dog("Pochi")
# pochi.name = "Pochi"
pochi.bark()

hachi = Dog("Hachi")
# hachi.name = "Hachi"
hachi.bark()

pochi.bark()