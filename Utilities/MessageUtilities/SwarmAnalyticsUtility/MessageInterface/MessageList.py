class MessageList():
    def __init__(self):
        self.Messages = []

    def addMessage(self,Message):
        comID = Message.CommunicationID
        stepID = self.NumberOfMessages(comID) + 1
        Message.SetStepID(stepID)

        self.Messages.append(Message)

    def GetLastNMessages(self,N,CommunicationID = None):
        if CommunicationID is not None:
            msgs = [msg for msg in self.Messages if msg.CommunicationID == CommunicationID]
        else:
            msgs = self.Messages
        return msgs[len(msgs)-N:]

    def NumberOfMessages(self,CommunicationID = None):
        if CommunicationID is not None:
            msgs = [msg for msg in self.Messages if msg.CommunicationID == CommunicationID]
        else:
            msgs = self.Messages
        return len(msgs)