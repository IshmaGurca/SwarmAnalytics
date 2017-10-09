from socketIO_client import SocketIO, BaseNamespace
from SwarmAnalyticsUtility.MessageInterface import MessageInterface
from uuid import uuid4

class SocketIOClient():

    _namespaces = []

    def __init__ (self, NameSpaces):
        self.MessageSenderID = uuid4()
        #get  socketserver and PORT from ENV variable!!!
        self.SocketIO  = SocketIO('socketioserver', 5000)
      
        for name in NameSpaces:
           self._namespaces.append({'Name':name,'Class':self.SocketIO.define(BaseNamespace,'/' + name)})
        #self.SocketIO.wait()


    #listen to all Namespaces. List of handler function needs to be in same  order as prior NameSpace names
    def listen(self, handlerList):
        self._handlerList = handlerList
        #for name in [x['Name'] for x in self._namespaces]:
        for name in [x for x in self._namespaces]:
            self.listenToNameSpace(name['Name'],handlerList[self._namespaces.index(name)])

        self.SocketIO.wait(seconds=600)
        

    def listenToNameSpace(self,NameSpaceName,handler, event = 'mymessage'):
        #get NameSpaceClass
        NameSpaceClass = [x['Class'] for x in self._namespaces if x['Name'] == NameSpaceName][0]
        NameSpaceClass.on(event, handler)

    def stoplistening(self, NameSpaceName = None, event = 'mymessage'):
        if NameSpaceName is None:
            NameSpacesClasses = [x['Class'] for x in self._namespaces]
        else:
            NameSpacesClasses = [x['Class'] for x in self._namespaces if x['Name'] == NameSpaceName]

        for NameSpaceClass in NameSpacesClasses:
            NameSpaceClass.off(event) 

    def emit(self,Message, NameSpaceName, event='mymessage'):
        NameSpaceClass = [x['Class'] for x in self._namespaces if x['Name'] == NameSpaceName][0]
        #print('EmitMessage: '+ str(Message.to_document()))
        NameSpaceClass.emit(event,Message.to_document())


        