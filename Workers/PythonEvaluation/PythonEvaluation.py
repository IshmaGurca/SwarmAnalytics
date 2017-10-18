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
import inspect

import Executables


class PythonEvaluation():
    def __init__ (self):
        self._socketIONamespaces = ['PythonEvaluation']
        self._socketIOSenderID = uuid4()
        self._socketIOClient = SocketIOClient(self._socketIONamespaces)
        self.MessageType = MessageType()
        self.RewardClass = FeedbackReward()
        self.MessageText = MessageText()
        self.RewardBounds = [-100,100]
        # make a list of safe functions
        self.safe_list = inspect.getmembers(Executables,inspect.isfunction)
        #self.safe_dict = dict([ (k, locals().get(k, None)) for k in safe_list ])
        self.safe_dict = dict(self.safe_list)
        # add any needed builtins back in
        #self.safe_dict['len'] = len

    
    def On_PythonEvaluation_Message(self,data):
        msg = MessageInterface.from_document(data)
        comID = msg.CommunicationID
        doAction = True
        if msg.SenderID != str(self._socketIOSenderID):
            if msg.MessageType == self.MessageType.Enum.QUESTION:
            #convert messahe to text:
                command = self.MessageText.IndexToText(msg.Data)
                self.act(command,comID)
            else:
                self.canthandle(comID)
    


    def ListenToSocketIO(self):
        self._socketIOHandler = [self.On_PythonEvaluation_Message]
        self._socketIOClient.listen(self._socketIOHandler)


    def act(self,command,CommunicationID):
        try:
            result = eval(command, {"__builtins__" : None }, self.safe_dict)
            if result != str:
                result = str(result)
            reward = -1
        except:
            result = 'ERROR'
            reward = -10

        #convert reward to new bounds
        reward = self.RewardClass.NormalizeToBounds([reward],self.RewardBounds)[0]

        self.emitObservation(result,reward, False, CommunicationID)

    def canthandle(self,comID):
        ob = '?'
        reward = -50
        self.emitObservation(ob,reward,False,comID)

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
        