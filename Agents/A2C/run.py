#!/usr/bin/env python
import os, logging, gym
from baselines import logger
from baselines.a2c.a2c import learn
from baselines.a2c.policies import  MyPolicy

try:
    from SwarmAnalyticsUtility.MessageInterface import MESSAGETYP, MessageInterface
    from SwarmAnalyticsUtility.CommunicationEnviroment import CommunicationEnviroment
except ImportError:
    import pip
    pip.main(['install','-e','/shared/MessageUtilities'])
    #print('Pip install SwarmAnalyticsUtility')
    from SwarmAnalyticsUtility.MessageInterface import MESSAGETYP, MessageInterface
    from SwarmAnalyticsUtility.CommunicationEnviroment import CommunicationEnviroment



def train(policy, lrschedule):
    
    '''
    Namespaces as dictionary:
        - ChatType e.g. 
            AGENT
            WORKER
            KNOWLEDGEBASE
        - NamespaceNames e.g. ['Client1', 'Client2']
    '''
    socketIONamespaces = {'AGENT':['OpenAIGym'],'WORKER':[], 'KNOWLEDGEBASE':[]}

    env = CommunicationEnviroment(socketIONamespaces)
    if policy == 'my':
        policy_fn = MyPolicy
    learn(policy_fn, env, lrschedule=lrschedule)

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--policy', help='Policy architecture', choices=['my'], default='my')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    args = parser.parse_args()
    print('Start Agent')
    train(policy=args.policy, lrschedule=args.lrschedule)

if __name__ == '__main__':
    main()
