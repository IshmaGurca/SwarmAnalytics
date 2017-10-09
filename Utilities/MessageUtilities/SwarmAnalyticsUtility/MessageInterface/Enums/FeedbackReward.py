from enum import Enum
import numpy as np

class FeedbackRewardSettings:
    lowerBound = -100
    upperBound = 100
    #List = [i for i in range(lowerBound,upperBound+1)]
   


class FeedbackReward:

    def __init__(self):    
        self.Settings = FeedbackRewardSettings
        self.n = self.Settings.upperBound - self.Settings.lowerBound + 1

    def IndexToOneHot(self, indices):
        import tensorflow as tf
        return tf.one_hot(indices,self.n)

    def OneHotToIndex(self,onehots):
        import tensorflow as tf
        return tf.argmax(onehots,axis = 1)

    def IndexToValue(self,indices):
        return [i + self.Settings.lowerBound for i in indices]

    def ValueToIndex(self, values):
        return np.asarray([v - self.Settings.lowerBound for v in values])

    def ValueToOneHot(self,values):
        indices = self.ValueToIndex(values)
        return self.IndexToOneHot(indices)

    def OneHotToValue(self, onehots):
        indices = self.OneHotToIndex(onehots)
        return self.IndexToValue(indices)

    def NormalizeValue(self,values,to=[-1,1]):
        oL = self.Settings.lowerBound
        oU = self.Settings.upperBound
        nL = to[0]
        nU = to[1]
        return [((v - oL)/(oU-oL))*(nU-nL) + nL  for v in values]

    def NormalizeToBounds(self,values, from_bound=[-1,1]):
        oL = from_bound[0]
        oU = from_bound[1]
        nL = self.Settings.lowerBound
        nU = self.Settings.upperBound
        return [((v - oL)/(oU-oL))*(nU-nL) + nL  for v in values]


if __name__ == "__main__":
    x = FeedbackReward()
    print(x.NormalizeValue([-100,-50,20,60,100,0]))
    print(x.NormalizeValue([-100,-50,20,60,100,0],[0,2]))
    print('Hello')