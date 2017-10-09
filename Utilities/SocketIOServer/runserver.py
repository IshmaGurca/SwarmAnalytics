"""
This script runs the MyHomePage_Flask application using a development server.
"""

from Server import app, socketio

if __name__ == '__main__':
    HOST = '0.0.0.0'
    PORT = 5000
    socketio.run(app,HOST,debug = False, log_output = False)