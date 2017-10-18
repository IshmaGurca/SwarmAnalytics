from enum import Enum
import random

class Calculation():

    def __init__(self):
        self.numbers = NUMBERS
        self.operations = OPERATIONS

    def RandomQA(self):
        n1 = random.randrange(0,len(self.numbers))
        n2 = random.randrange(0,len(self.numbers))
        op = random.randrange(0,len(self.operations))

        Q = self.GetQuestion(n1,op,n2)
        A = self.GetAnswer(n1,op,n2)
        
        return (Q,A)

    def GetQuestion(self,n1,op, n2):
        n1txt = self.numbers(n1).name
        n2txt = self.numbers(n2).name

        opEnum = self.operations(op)
        optxt = ""
        if opEnum == self.operations.MINUS:
            optxt =  "minus"
        elif opEnum == self.operations.TIMES:
            optxt = "times"
        elif opEnum == self.operations.DIVIDED:
            optxt = "divided by"
        else:
            optxt = "plus"
        
        txt = "What is " + n1txt + " " + optxt  + " " +  n2txt+ "?"
        return txt

    def GetAnswer(self,n1,op, n2):
        opEnum = self.operations(op)
        res = 0
        un = ""
        if opEnum == self.operations.MINUS:
            res = n1 - n2
        elif opEnum == self.operations.TIMES:
            res = n1 * n2
        elif opEnum == self.operations.DIVIDED:
            if n2 == 0:
                un = "undefined"
            else:
                res = n1 / n2
        else:
            res  =n1 + n2

        if un != "":
            ret = un
        else: 
            ret =  str(int(res))
        return ret


class NUMBERS(Enum):
    zero = 0
    one = 1
    two = 2
    three = 3
    four = 4
    five = 5
    six = 6
    seven = 7
    eight = 8
    nine = 9
    ten = 10 

class OPERATIONS(Enum):
    PLUS = 0
    MINUS = 1
    TIMES = 2
    DIVIDED = 3

if __name__ == "__main__":
    c = Calculation()
    print(c.RandomQA())
