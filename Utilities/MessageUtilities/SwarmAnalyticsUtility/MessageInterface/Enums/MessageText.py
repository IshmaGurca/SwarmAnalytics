from enum import Enum
import numpy as np

class MessageTextSettings:
    encoding = 'ascii'
    NofChars = 128
    #List = [i for i in range(lowerBound,upperBound+1)]
   
class MessageTextAdditionalEnum(Enum):
    STARTMESSAGE = 0
    ENDMESSAGE = 1

class MessageText:

    def __init__(self):
        self.Additions = MessageTextAdditionalEnum
        self.Settings = MessageTextSettings
        self.n = len(self.Additions) + self.Settings.NofChars

    def IndexToOneHot(self, indices):
        import tensorflow as tf
        return tf.one_hot(indices,self.n)

    def OneHotToIndex(self,onehots):
        import tensorflow as tf
        return tf.argmax(onehots,axis = 1)

    def IndexSartMessage(self,times):
        return np.asarray([self.Additions.STARTMESSAGE.value]*times)

    def IndexToText(self,indices, ReturnPureText = True):
        text = []
        for ind in indices:
            t = []
            startj = 0
            endj = len(ind)
            for j,i in enumerate(ind):
                if i not in range(len(self.Additions)):
                    if ReturnPureText and j > startj and j < endj:
                    #t.append(chr(i - len(self.Additions)).decode(self.Settings.encoding))
                        t.append(chr(i - len(self.Additions)))
                else:
                    if self.Additions(i) == self.Additions.STARTMESSAGE:
                        startj = j
                    if self.Additions(i) == self.Additions.ENDMESSAGE:
                        endj = j

                if ReturnPureText == False and i in range(len(self.Additions)):
                    t.append(self.Additions(i))
            text.append(''.join(t))
        return text

    def TextToIndex(self, txts, AddStartEndPerTxt = True):
        ind = []
        for txt in txts:
            i = []
            if  AddStartEndPerTxt:
                i = i + [self.Additions.STARTMESSAGE.value]

            i = i + [ord(t.encode(self.Settings.encoding)) + len(self.Additions) for t in txt]
            
            if AddStartEndPerTxt:
                i = i +[self.Additions.ENDMESSAGE.value]
            ind.append(i)
        return np.asarray(ind)

    def TextToOneHot(self,txts):
        indices = self.TextToIndex(txts)
        return self.IndexToOneHot(indices)

    def OneHotToText(self, onehots):
        indices = self.OneHotToIndex(onehots)
        return self.IndexToText(indices)

    def PadIndex(self,indices, maxlength):
        #print(type(indices))
        ENDMESSAGE_index = self.Additions.ENDMESSAGE.value
        n_indices = []
        for i,ind in enumerate(indices):
            #print(type(ind))
            #indices[i] = ind + [1]         
            if    len(ind) < maxlength:
                n_indices.append(np.lib.pad(ind,(0,maxlength-len(ind)), 'constant', constant_values=(ENDMESSAGE_index, ENDMESSAGE_index)).tolist())
            else:
                n_indices.append(ind.tolist()[:maxlength])

        return np.asarray(n_indices)

    def MaskEndMessage(self, indices):
        import tensorflow as tf
        end = tf.zeros(indices.shape,tf.int64) + self.Additions.ENDMESSAGE.value
        return tf.to_int64(tf.not_equal(indices, end))
        #return [int(self.Additions.ENDMESSAGE.value != i) for i in indices]


if __name__ == "__main__":
    x = MessageText()
    text = ['[1.2334,4.3356]','Selber Hallo']
    i = x.TextToIndex(text)
    print(i)
    i =x.PadIndex(i,20)
    print(i)
    print(type(i))
    y = [[88,0,93,51,48,52,1,43,43,4,4,4]]
    t = x.IndexToText(i)
    print(t)
    print('Hello')