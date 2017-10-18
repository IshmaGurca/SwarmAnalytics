try:
    from SwarmAnalyticsUtility.SocketIOClient import SocketIOClient
    from SwarmAnalyticsUtility.MessageInterface import  MessageInterface
    from SwarmAnalyticsUtility.MessageInterface.Enums import FeedbackReward, MessageType
except ImportError:
    import pip
    pip.main(['install','-e','/shared/MessageUtilities'])
    #print('Pip install SwarmAnalyticsUtility')
    from SwarmAnalyticsUtility.SocketIOClient import SocketIOClient
    from SwarmAnalyticsUtility.MessageInterface import MessageInterface
    from SwarmAnalyticsUtility.MessageInterface.Enums import FeedbackReward, MessageType
import gym
from uuid import uuid4
import time
import numpy as np

class OpenAIGym():
    def __init__ (self):
        self._socketIONamespaces = ['OpenAIGym']
        self._socketIOSenderID = uuid4()
        self.env = gym.make('Copy-v0')
        self._socketIOClient = SocketIOClient(self._socketIONamespaces)
        self._actionSpace = self.env.action_space
        self.MessageType = MessageType()
        self.RewardClass = FeedbackReward()
        self.RewardBounds = [-1,1]
    
        time.sleep(10)
        self.reset()

    
    def On_OpenAIGym_Message(self,data):
        msg = MessageInterface.from_document(data)
        comID = msg.CommunicationID
        doAction = True
        if msg.MessageType == self.MessageType.Enum.ANSWER:
            actionRaw = [int(a) for a in msg.Data]
            if type(self.env.action_space.sample()) == tuple:
                n = len(self.env.action_space.sample())
                action = actionRaw[:n]
                if self.env.action_space.contains(action) == False:
                    doAction = False
            else:

                if int(actionRaw[0]) in range(self.env.action_space.n):
                    action = int(actionRaw[0])
                else:
                    doAction = False

            if doAction:
                self.act(action, comID)
            else:
                self.canthandle(comID)
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
        msgTyp = self.MessageType.Enum.OBERSAVATION
        nameSpace = self._socketIONamespaces[0]
        #convert observation from numpy to list and then to string
        #print(Observation)
        if type(Observation) != str:
            if type(Observation)  == int:
                Observation = str(Observation)
            else:
                Observation = str(list(Observation))
        obBody = Observation
        Message = MessageInterface(nameSpace,self._socketIOSenderID, msgTyp,CommunicationID,Data = obBody, Reward = Reward, DoneFlag = Done)
        self._socketIOClient.emit(Message,nameSpace)
        #print('sendObservation')
        
    
        