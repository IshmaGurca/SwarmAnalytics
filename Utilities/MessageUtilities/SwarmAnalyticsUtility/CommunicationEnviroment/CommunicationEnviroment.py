from SwarmAnalyticsUtility.SocketIOClient import SocketIOClient
from SwarmAnalyticsUtility.MessageInterface.MessageInterface import MESSAGETYP, MessageInterface
from SwarmAnalyticsUtility.MessageInterface.MessageList import MessageList
from uuid import uuid4
import numpy as np
from  SwarmAnalyticsUtility.MessageInterface.Enums import ChatType, ChatNumber, MessageType, MessageText, FeedbackReward


class CommunicationEnviroment():
    def __init__(self,SocketIONamespaces, CallbackFunction = None):
        ''' 
        Namespaces as dictionary:
        - ChatType e.g. 
            AGENT
            WORKER
            KNOWLEDGEBASE
        - NamespaceNames e.g. ['Client1', 'Client2']
        '''
        self._socketIONamespacesDict = SocketIONamespaces
        self._socketIONamespaces = [n for c  in self._socketIONamespacesDict.values() for n in c]
        self._socketIOSenderID = uuid4()
        self._socketIOClient = SocketIOClient(self._socketIONamespaces)
        self._messageCallback = CallbackFunction
        self.Messages = MessageList()

        self.ChatType = ChatType()
        self.ChatNumber = ChatNumber()
        self.MessageType = MessageType()
        self.MessageText = MessageText()
        self.FeedbackReward = FeedbackReward()
        self.LengthOfMessageText = MessageInterface.MaxLengthOfMessageBody
        self.ObsSpace = [1,1,1,self.LengthOfMessageText]
        self.ActSpace = [1,1,1,self.LengthOfMessgeText,1] 

    def SetCallbackFunction(CallbackFunction):
        self._messageCallback = CallbackFunction

    def ListenToSocketIO(self):

        '''
        DEBUG emit
        
        iNAMESPACE = 0
        iMESSAGETYP = 1
        body = 'Hello'
        self.emit(uuid4(),iNAMESPACE,iMESSAGETYP,body)
        '''

        def On_Message(data):
            msg = MessageInterface.from_document(data)
            if msg.SenderID != str(self._socketIOSenderID):
                self.Messages.addMessage(msg)
                msg = self.Messages.GetLastNMessages(1,msg.CommunicationID)[0]

                observation, reward, done, comID, step = self.MsgToEnv(msg)
                self._messageCallback(observation, reward, done, comID, step)
            
        self._socketIOHandler = [On_Message] # [On_Message for i in range(len(self._socketIONamespaces))]
        print('Start listenning')
        self._socketIOClient.listen(self._socketIOHandler)

    def MsgToEnv(self, msg):
        comID = msg.CommunicationID
        step = msg.StepID
        #generate observation
        '''
        ChatType
        ChatNr
        MessageType
        MessageText

        '''
        nspace = msg.NameSpace
        chatDetail = [(k,v.index(nspace)) for k,v in self._socketIONamespacesDict.items() if nspace in v]
        ctyi = self.ChatType.EnumToIndex([self.ChatType.GetEnumByName(chatDetail[0][0])])
        cnri = self.ChatNumber.EnumToIndex([self.ChatNumber.GetEnumByIndex(chatDetail[0][1])])

        mtyi = self.MessageType.EnumToIndex([msg.MessageType])
        mtxt = self.MessageText.TextToIndex([msg.data])
        mtxt = self.MessageText.PadIndex(mtxt,self.LengthOfMessageText)

        observation = np.concatenate(ctyi,cnri,mtyi,mtxt)
        reward = msg.Reward
        done = msg.DoneFlag

        return observation, reward, done, comID, step


    def emit(self,CommunicationID,action):
        ctyi,cnri, mtyi, mtxt, reward = np.split(action,np.cumsum(self.ActSpace))[:-1]
        
        #What if namespace not exists?????
        nameSpace = self._socketIONamespacesDict[self.ChatType.GetEnumNameByValue(ctyi)][cnri]
        msgTyp = self.MessageType.IndexToEnum(mtyi)[0]
        if isinstance(mtxt, np.ndarray):
            bmtxtody = list(mtxt) 
        #a  = [0.0,1.1]
        #a[0] = body[0]
        obBody =[float(mtxt)]
        
        Message = MessageInterface(nameSpace,self._socketIOSenderID, msgTyp,CommunicationID,Data = obBody, Reward = reward[0])
        self._socketIOClient.emit(Message,nameSpace)
