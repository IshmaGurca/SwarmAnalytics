"""
This script runs the OpenAI_Gym Client on an infinit loop.
""" 

from PythonEvaluation import PythonEvaluation

def start():
    print('Start')
    #Init the OpenAI_Gym Client   
    client = PythonEvaluation()
    client.ListenToSocketIO()
    print('End')
    #client._socketIOClient.SocketIO.wait()

if __name__ == '__main__':
    start()