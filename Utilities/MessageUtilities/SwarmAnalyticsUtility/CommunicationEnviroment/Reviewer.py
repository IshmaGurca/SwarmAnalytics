try:
    from SwarmAnalyticsUtility.MessageInterface import  MessageInterface
except ImportError:
    import pip
    pip.main(['install','-e','/shared/MessageUtilities'])
    from SwarmAnalyticsUtility.MessageInterface import MessageInterface

import sys
class Reviewer():

    FAIL_REWARD_MAX = -50
    FAIL_REWARD_MIN = -100
    USED_NAMESPACE = 'InternReview'
    FAILS = 0

    

    def __init__(self, CommunicationEnv):
        self.CommunicationEnv = CommunicationEnv
        sys.setrecursionlimit(1500)

    def review(self, message):
        result = True
        for fn in self.CommunicationEnv._ReviewFunctions:
            if fn(message) == False:
                result = False
                self.FAILS += 1
                self.emitReview(message)

        return result


    def emitReview(self, message):
        nameSpace = self.USED_NAMESPACE
        reward =  max(self.FAIL_REWARD_MIN,self.FAIL_REWARD_MAX + int(self.FAILS * (self.FAIL_REWARD_MIN - self.FAIL_REWARD_MAX)))
        obBody = 'failed review'   
        msgTyp = self.CommunicationEnv.MessageType.Enum.OBERSAVATION
        CommunicationID = message.CommunicationID
        senderID = self.CommunicationEnv._socketIOSenderID

        msg = MessageInterface(nameSpace,senderID, msgTyp,CommunicationID,Data = obBody, Reward = reward, DoneFlag = False)
        self.CommunicationEnv._socketIOClient.emit(msg,nameSpace)
        
        
        #self.CommunicationEnv.Messages.addMessage(msg)
        #msg = self.CommunicationEnv.Messages.GetLastNMessages(1,msg.CommunicationID)[0]
        #observation, reward, done, comID, step = self.CommunicationEnv.MsgToEnv(msg)

        #self.CommunicationEnv._messageCallback(observation, reward, done, comID, step)