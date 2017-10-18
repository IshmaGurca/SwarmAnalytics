try:
    from SwarmAnalyticsUtility.SocketIOClient import SocketIOClient
    from SwarmAnalyticsUtility.MessageInterface import  MessageInterface
    from SwarmAnalyticsUtility.MessageInterface.Enums import FeedbackReward, MessageType, MessageText
except ImportError:
    import pip
    pip.main(['install','-e','/shared/MessageUtilities'])
    #print('Pip install SwarmAnalyticsUtility')
    from SwarmAnalyticsUtility.SocketIOClient import SocketIOClient
    from SwarmAnalyticsUtility.MessageInterface import MessageInterface
    from SwarmAnalyticsUtility.MessageInterface.Enums import FeedbackReward, MessageType, MessageText
from uuid import uuid4
import time

from QAGenerators.Calculation import Calculation

import difflib


class QATrainer():

    MAX_WRONGANSWERS = 30

    def __init__ (self):
        self._socketIONamespaces = ['QATrainer']
        self._socketIOSenderID = uuid4()
        self._socketIOClient = SocketIOClient(self._socketIONamespaces)
        self.MessageType = MessageType()
        self.RewardClass = FeedbackReward()
        self.MessageText = MessageText()
        self.RewardBounds = [-100,100]
        self.QAGenerators = [Calculation()]
        self.ExpectedAnswer = ""
        self.CurrentQuestion = ""
        time.sleep(30)
        print('start questions')
        self.reset()
        self.wrongAnswers = 0

    
    def On_QATrainer_Message(self,data):
        msg = MessageInterface.from_document(data)
        comID = msg.CommunicationID
        doAction = True
        #print(msg.MessageType)
        if msg.SenderID != str(self._socketIOSenderID):
            
            if msg.MessageType == self.MessageType.Enum.ANSWER:
            #convert messahe to text:
                answer = msg.Data #self.MessageText.IndexToText(msg.Data)
                self.act(answer,comID)
            else:
                self.canthandle(comID)
    


    def ListenToSocketIO(self):
        self._socketIOHandler = [self.On_QATrainer_Message]
        self._socketIOClient.listen(self._socketIOHandler)


    def act(self,answer,CommunicationID):
        #print(answer)
        
        seq = difflib.SequenceMatcher(a=self.ExpectedAnswer.lower(), b=answer.lower())
        seq.ratio()

        #if answer == self.ExpectedAnswer:
        if seq.ratio() == 1:
            ob = "Correct"
            done = True
            reward = 100
        else:
            ob = self.CurrentQuestion
            done = False
            re = seq.ratio()*100
            reward = int(re)
            self.wrongAnswers = self.wrongAnswers + 1
    
        #convert reward to new bounds
        reward = self.RewardClass.NormalizeToBounds([reward],self.RewardBounds)[0]

        if self.wrongAnswers >= self.MAX_WRONGANSWERS:
            done = True
            reward = -100

        self.emitObservation(ob,reward, done, CommunicationID)
        if done:
            self.reset()

    def reset(self):
        comID = uuid4()
        qa = self.QAGenerators[0].RandomQA()
        self.ExpectedAnswer = qa[1]
        ob  = qa[0]
        self.wrongAnswers = 0
        print(ob)
        self.CurrentQuestion = ob 
        self.emitObservation(ob, Reward=0.0, Done = False,CommunicationID = comID)

    def canthandle(self,comID):
        done = False
        if self.wrongAnswers >= self.MAX_WRONGANSWERS:
            done = True
            reward = -100
        else:
            reward = -50    
            self.wrongAnswers = self.wrongAnswers + 1
        
        ob = self.CurrentQuestion
        self.emitObservation(ob,reward,done,comID)
        if done:
            self.reset()

    def emitObservation(self,Observation,Reward, Done,CommunicationID):
        msgTyp = self.MessageType.Enum.OBERSAVATION
        nameSpace = self._socketIONamespaces[0]
        #convert observation from numpy to list and then to string
        if type(Observation) != str:
            if type(Observation)  == int:
                Observation = str(Observation)
            else:
                Observation = str(list(Observation))
        obBody = Observation
        Message = MessageInterface(nameSpace,self._socketIOSenderID, msgTyp,CommunicationID,Data = obBody, Reward = Reward, DoneFlag = Done)
        self._socketIOClient.emit(Message,nameSpace)  
        