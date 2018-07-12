import math

# vector model
def model(lis):
    sum = 0.0
    for each in lis:
        sum += float(each)*float(each)
    return math.sqrt(sum)

# vector dot
def dot(lis1, lis2):
    sum = 0.0
    for a, b in zip(lis1, lis2):
        sum += float(a) * float(b)
    return sum

def cos(lis1, lis2, mod1 = 0, mod2 = 0):
    if mod1 == 0:
        mod1 = model(lis1)
    if mod2 == 0:
        mod2 = model(lis2)
    return dot(lis1, lis2)/(mod1*mod2)