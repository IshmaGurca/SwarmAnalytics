from enum import Enum
import numpy as np

class ChatNumberEnum(Enum):
    CHATNR0 = 0
    CHATNR1 = 1
    CHATNR2 = 2
    CHATNR3 = 3
    CHATNR4 = 4
    CHATNR5 = 5


class ChatNumber:

    def __init__(self):    
        self.Enum = ChatNumberEnum
        self.n = len(self.Enum)

    def IndexToOneHot(self, indices):
        import tensorflow as tf
        return tf.one_hot(indices,self.n)

    def OneHotToIndex(self,onehots):
        import tensorflow as tf
        return tf.argmax(onehots,axis = 1)

    def IndexToEnum(self,indices):
        return [x.Enum(i) for i in indices]

    def EnumToIndex(self, enums):
        return np.asarray([e.value for e in enums])

    def EnumToOneHot(self,enums):
        indices = self.EnumToIndex(enums)
        return self.IndexToOneHot(indices)

    def OneHotToEnum(self, onehots):
        indices = self.OneHotToIndex(onehots)
        return self.IndexToEnum(indices)

    def GetEnumByIndex(self,index):
        return self.Enum(index)

if __name__ == "__main__":
    x = ChatNumberEnum()
    print(x.Enum.ANSWER)
    print(x.Enum(0))
    print('Hello')