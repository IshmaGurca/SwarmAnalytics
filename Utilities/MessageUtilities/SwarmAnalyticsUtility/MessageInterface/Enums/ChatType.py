from enum import Enum
import numpy as np

class ChatTypeEnum(Enum):
    AGENT = 0
    WORKER = 1
    KNOWLEDGEBASE = 2
    REVIEW = 3


class ChatType:

    def __init__(self):    
        self.Enum = ChatTypeEnum
        self.n = len(self.Enum) - 1

    def IndexToOneHot(self, indices):
        import tensorflow as tf
        return tf.one_hot(indices,self.n)

    def OneHotToIndex(self,onehots):
        import tensorflow as tf
        return tf.argmax(onehots,axis = 1)

    def IndexToEnum(self,indices):
        return [self.Enum(i) for i in indices]

    def EnumToIndex(self, enums):
        return np.asarray([e.value for e in enums])

    def EnumToOneHot(self,enums):
        indices = self.EnumToIndex(enums)
        return self.IndexToOneHot(indices)

    def OneHotToEnum(self, onehots):
        indices = self.OneHotToIndex(onehots)
        return self.IndexToEnum(indices)

    def GetEnumByName(self, name):
        return self.Enum[name]

    def GetEnumNameByValue(self, value):
        return self.Enum(value).name



if __name__ == "__main__":
    x = ChatTypeEnum()
    print(x.Enum.ANSWER)
    print(x.Enum(0))
    print('Hello')