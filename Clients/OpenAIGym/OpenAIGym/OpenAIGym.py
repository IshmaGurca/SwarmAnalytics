try:
    from SwarmAnalyticsUtility.SocketIOClient import SocketIOClient
    from SwarmAnalyticsUtility.MessageInterface import MESSAGETYP, MessageInterface
    from SwarmAnalyticsUtility.MessageInterface.Enum import FeedbackReward
except ImportError:
    import pip
    pip.main(['install','-e','/shared/MessageUtilities'])
    #print('Pip install SwarmAnalyticsUtility')
    from SwarmAnalyticsUtility.SocketIOClient import SocketIOClient
    from SwarmAnalyticsUtility.MessageInterface import MESSAGETYP, MessageInterface
    from SwarmAnalyticsUtility.MessageInterface.Enum import FeedbackReward
import gym
from uuid import uuid4
import time
import numpy as np

class OpenAIGym():
    def __init__ (self):
        self._socketIONamespaces = ['OpenAIGym']
        self._socketIOSenderID = uuid4()
        self.env = gym.make('CartPole-v0')
        self._socketIOClient = SocketIOClient(self._socketIONamespaces)
        self._actionSpace = self.env.action_space
        self.RewardClass = FeedbackReward()
        self.RewardBounds = [-1,1]
    
        time.sleep(10)
        self.reset()

    
    def On_OpenAIGym_Message(self,data):
        msg = MessageInterface.from_document(data)
        comID = msg.CommunicationID
        if msg.MessageTyp == MESSAGETYP.ANSWER:
            action = msg.Data
            action = int(action[0])
            self.act(action, comID)

        else:
            self.canthandle(comID)
    


    def ListenToSocketIO(self):
        self._socketIOHandler = [self.On_OpenAIGym_Message]
        self._socketIOClient.listen(self._socketIOHandler)


    def act(self,action,CommunicationID):
        #print(str(self._actionSpace.contains(action)))
        #print(str(action))
        ob, reward, done, _ = self.env.step(action)

        #convert reward to new bounds
        reward = self.RewardClass.NormalizeToBounds([reward],self.RewardBounds)[0]

        self.emitObservation(ob,reward, done, CommunicationID)
        if done:
            self.reset()

    def reset(self):
        comID = uuid4()
        #self.emitReset(comID)
        ob = self.env.reset()
        self.emitObservation(ob, Reward=0.0, Done = False,CommunicationID = comID)

    def canthandle(self,comID):
        ob = '?'
        reward = -50
        self.emitObservation(ob,reward,False,comID)

    def emitObservation(self,Observation,Reward, Done,CommunicationID):
        msgTyp = MESSAGETYP.OBERSAVATION
        nameSpace = self._socketIONamespaces[0]
        #convert observation from numpy to list and then to string
        if type(Observation) != str:
            Observation = str(list(Observation))
        obBody = Observation
        Message = MessageInterface(self._socketIOSenderID, msgTyp,CommunicationID,Data = obBody, Reward = Reward, DoneFlag = Done)
        self._socketIOClient.emit(Message,nameSpace)
        #print('sendObservation')
        
    
        