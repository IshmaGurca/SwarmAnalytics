from SwarmAnalyticsUtility.SocketIOClient import SocketIOClient
from SwarmAnalyticsUtility.MessageInterface.MessageInterface import MessageInterface
from SwarmAnalyticsUtility.MessageInterface.MessageList import MessageList
from uuid import uuid4
import numpy as np
from  SwarmAnalyticsUtility.MessageInterface.Enums import ChatType, ChatNumber, MessageType, MessageText, FeedbackReward

from SwarmAnalyticsUtility.CommunicationEnviroment.Reviewer import Reviewer

import os

class CommunicationEnviroment():
    def __init__(self,SocketIONamespaces, CallbackFunction = None, ReviewFunctionList = None):
        ''' 
        Namespaces as dictionary:
        - ChatType e.g. 
            AGENT
            WORKER
            KNOWLEDGEBASE
        - NamespaceNames e.g. ['Client1', 'Client2']
        '''
        #self._socketIONamespacesDict = SocketIONamespaces

        #GET NAMESACE dictionary from enviroment variables
        agent_ns = [ns for ns in os.getenv('AGENT_NAMESPACES').split(';')] if os.getenv('AGENT_NAMESPACES') is not None else []
        worker_ns = [ns for ns in os.getenv('WORKER_NAMESPACES').split(';')] if os.getenv('WORKER_NAMESPACES') is not None else []
        knowledge_ns = [ns for ns in os.getenv('KNOWLEDGEBASE_NAMESPACES').split(';')] if os.getenv('KNOWLEDGEBASE_NAMESPACES') is not None else []

        self._socketIONamespacesDict = {'AGENT':agent_ns,'WORKER':worker_ns, 'KNOWLEDGEBASE':knowledge_ns}
        #self._socketIONamespaces = [n for c  in self._socketIONamespacesDict.values() for n in c]
        #Add intern Review NameSapce
        self._socketIONamespacesDict['REVIEW'] = ['InternReview']
        self._socketIONamespaces = [n for c  in self._socketIONamespacesDict.values() for n in c]
        print(self._socketIONamespacesDict)

        #self._InternNamespacesDict = self._socketIONamespacesDict
        #self._InternNamespacesDict['REVIEW'] = 'InternReview'
        #self._InternNamespaces = [n for c  in self._InternNamespacesDict.values() for n in c]

        self._socketIOSenderID = uuid4()
        self._socketIOClient = SocketIOClient(self._socketIONamespaces)
        self._messageCallback = CallbackFunction
        self._ReviewFunctions = ReviewFunctionList
        self.Messages = MessageList()

        self.ChatType = ChatType()
        self.ChatNumber = ChatNumber()
        self.MessageType = MessageType()
        self.MessageText = MessageText()
        self.FeedbackReward = FeedbackReward()
        self.LengthOfMessageText = MessageInterface.MaxLengthOfMessageBody
        self.ObsSpace = [1,1,1,self.LengthOfMessageText]
        self.ActSpace = [1,1,1,self.LengthOfMessageText,1]

        self.BlackList = []

        def StandardReview(message):
            ns = message.NameSpace
            if ns in self._socketIONamespaces:
                return True
            else:
                return False


        self.Reviewer = Reviewer(self)
        self._ReviewFunctions = [StandardReview]

    def SetCallbackFunction(self,CallbackFunction):
        self._messageCallback = CallbackFunction

    def SetReviewFunctions(self,ReviewFunctionList):
        self._ReviewFunctions = ReviewFunctionList

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
            if (msg.SenderID != str(self._socketIOSenderID) or msg.NameSpace == 'InternReview') and msg.CommunicationID not in self.BlackList:
                self.Messages.addMessage(msg)
                msg = self.Messages.GetLastNMessages(1,msg.CommunicationID)[0]

                observation, reward, done, comID, step = self.MsgToEnv(msg)
                #forward Done True input to all other channels:
                if done:
                    self.ForwardFeedback(msg)
                    self.BlackList.append(msg.CommunicationID)
                self._messageCallback(observation, reward, done, comID, step)
            
        self._socketIOHandler = [On_Message for i in range(len(self._socketIONamespaces))]
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
        #print(nspace)
        chatDetail = [(k,v.index(nspace)) for k,v in self._socketIONamespacesDict.items() if nspace in v]
        ctyi = self.ChatType.EnumToIndex([self.ChatType.GetEnumByName(chatDetail[0][0])])
        cnri = self.ChatNumber.EnumToIndex([self.ChatNumber.GetEnumByIndex(chatDetail[0][1])])


        mtyi = self.MessageType.EnumToIndex([msg.MessageType])
        mtxt = self.MessageText.TextToIndex([msg.Data])
        mtxt = self.MessageText.PadIndex(mtxt,self.LengthOfMessageText)


        #if nspace == 'TinaBob':
        #    print(msg.Reward)

        ''' NOT same shape !!!!!!!'''
        mtxt = np.reshape(mtxt,(int(mtxt.shape[0])*int(mtxt.shape[1])))
        observation = np.concatenate((ctyi,cnri,mtyi,mtxt))
        reward = msg.Reward
        done = msg.DoneFlag
        return observation, reward, done, comID, step


    def emit(self,CommunicationID,action):
        ctyi,cnri, mtyi, mtxt, reward = np.split(action,np.cumsum(self.ActSpace), axis=1)[:-1]
        ctyi = ctyi[0]
        cnri = cnri[0][0]
        mtyi = mtyi[0]
        mtxt = mtxt[0]
        reward = reward[0]
        
        #What if namespace not exists?????
        # Reset Namespace to best possible e.g. nearest ctyi and nearest channel NR.
        nameSpace = ""
        try:
            nameSpace = self._socketIONamespacesDict[self.ChatType.GetEnumNameByValue(ctyi)][cnri]
        except:
            pass
        
        msgTyp = self.MessageType.IndexToEnum(mtyi)[0]
        if isinstance(mtxt, np.ndarray):
            mtxt = list(mtxt) 
        #a  = [0.0,1.1]
        #a[0] = body[0]
      
        #obBody = [float(m) for m in mtxt]
        obBody = self.MessageText.IndexToText([mtxt])[0]
        reward = self.FeedbackReward.IndexToValue(reward)

        #print(str(obBody))
        Message = MessageInterface(nameSpace,self._socketIOSenderID, msgTyp,CommunicationID,Data = obBody, Reward = float(reward[0])*0)
        if self.Reviewer.review(Message) == True:
            #print('EmitMessage: '+ str(Message.Reward))
            self._socketIOClient.emit(Message,nameSpace)

    def ForwardFeedback(self,msg):
        nspace = msg.NameSpace
        for ns in self._socketIONamespaces:
            if ns != nspace:
                Message = MessageInterface(ns,self._socketIOSenderID, msg.MessageType,msg.CommunicationID,Data = msg.Data, Reward = msg.Reward, DoneFlag = msg.DoneFlag)
                self._socketIOClient.emit(Message,ns)
