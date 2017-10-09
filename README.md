# SwarmAnalytics

Basic idea:

Communication between Client and Reinforcement Algorithm via SocketIO. This enables the process to run in seperate docker containers.

Current status:

First Prototype (NOT FUNCTIONAL)

Client: Open AI Gym enviroment
Agent: A2C RL implementation
Communication Server: Flask SocketIO



Info:

The current Agent is based on OPEN AI A2C implementation in the "Baselines" package. See https://github.com/openai/baselines