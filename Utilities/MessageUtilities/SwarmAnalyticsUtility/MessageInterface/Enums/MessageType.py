from enum import Enum
import numpy as np

class MessageTypeEnum(Enum):
    QUESTION = 0
    ANSWER = 1
    OBERSAVATION = 2


class MessageType:

    def __init__(self):    
        self.Enum = MessageTypeEnum
        self.n = len(self.Enum)

    def IndexToOneHot(self, indices):
        import tensorflow as tf
        return tf.one_hot(indices,self.n)

    def OneHotToIndex(self,onehots):
        import tensorflow as tf
        return tf.argmax(onehots,axis = 1)

    def IndexToEnum(self,indices):
        if type(indices) == np.ndarray:
            indices = indices.tolist()
        return [self.Enum(i) for i in indices]

    def EnumToIndex(self, enums):
        return np.asarray([e.value for e in enums])

    def EnumToOneHot(self,enums):
        indices = self.EnumToIndex(enums)
        return self.IndexToOneHot(indices)

    def OneHotToEnum(self, onehots):
        indices = self.OneHotToIndex(onehots)
        return self.IndexToEnum(indices)


if __name__ == "__main__":
    x = MessageType()
    print(x.Enum.ANSWER)
    print(x.Enum(0))
    print('Hello')