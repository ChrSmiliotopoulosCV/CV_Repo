# ############# Python Objects #############
# ################# Warning ################

# This lecture is much about definitions and mechanics for objects. 
# It is lot more about "how it works" and less about "how you use it". 
# You won't get the entire picture until this is all looked at in the 
# context of a real life problem. So please suspend disbelief and learn 
# technique for the next 50 or so slides.

# ################# Class ################
from unicodedata import name


class PartyAnimal:
    x = 0

    def party(self):
        self.x = self.x + 1
        print("So far ", self.x)

an = PartyAnimal()

# print("Type:", type(an))
# print("Dir:", dir(an))

an.party()
an.party()
an.party()

# ############ Playing with dir() and type() ############
# We use dir() and type() in order to inspect variables, 
# functions, types and objects.

# dir(): It is the command with which capabilities are listed. 
# The ones with underscores should be ignored because they are 
# used by Python itself.

x = list()
print("The type of x is ", type(x))
print("The available functions regarding x are ", dir(x))

# ############ Object Lifecycle ############
# Constructor: The primary purpose of the constructor is to set 
# up some instance variables to have the proper initial values 
# when the object is created.
# Destructors: We will see very little constructors, although they
# are use very rarely in Python. It is seldom used.

class PartyAnimal:
    x=0

    def __init__(self):
        print('I am constructed.')

    def party(self):
        self.x = self.x + 1
        print('So far ', self.x)

an = PartyAnimal()
an.party()
an.party()
an.party()
an = 42
print('an contains', an)

# ############ Many Instance of the Same Class ############
class PartyAnimal:
    x=0
    name = ""
    def __init__(self, z):
        self.name = z
        print(self.name, 'constructed.')

    def party(self):
        self.x = self.x + 1
        print(self.name, 'party count', self.x)

an = PartyAnimal("Christos")
an.party()

s = PartyAnimal("Maraki")
s.party()
an.party()

class FootballFan(PartyAnimal):
    points = 0
    def touchdown(self):
        self.points = self.points + 7
        self.party()
        print(self.name, "points", self.points)

j = FootballFan("Jim")
j.party()
j.touchdown()

# ################## Definitions ##################
# Class: a template
# Attribute: A variable within a class
# Method: A function within a class
# Object: A particular instance of a class
# Constructor: Code that runs when an object is created. 
# Inheritance: The ability to extend a class to make a new class.


