from Server import app, socketio
from flask_socketio import emit

@socketio.on('external')
@socketio.on('internal')
def callMsgHandler(messagelist):
    print('int..ext')
    #print(request.sid)
    #sid = request.sid
    #msgHandler.HandleMessageDoc(messagelist,sid)
    
@socketio.on('disconnect')
def client_disconnect():
    pass
    #sid = request.sid
    #print('Client disconnected with id:')
    #msgHandler.saveMessageHistory(sid)

@socketio.on('connect')
def client_disconnect():
    pass
    #sid = request.sid
    #print('Client connected with id:')
    #msgHandler.saveMessageHistory(sid)
    ###

''' @socketio.on('mymessage', namespace='/OpenAIGym')
def hanlde_mymessage(json):
    print('Message:' + str(json))
    emit('mymessage', {'data':'IS ME the server.'}, namespace='/OpenAIGym', broadcast=True) '''


@socketio.on('mymessage',namespace='/OpenAIGym')
def hanlde_mymessage(json):
    emit('mymessage',json,broadcast=True)


@socketio.on('mymessage',namespace='/QATrainer')
def hanlde_mymessage(json):
    emit('mymessage',json,broadcast=True)


@socketio.on('mymessage',namespace='/TinaBob')
def hanlde_mymessage(json):
    emit('mymessage',json,broadcast=True)


@socketio.on('mymessage',namespace='/PythonEvaluation')
def hanlde_mymessage(json):
    emit('mymessage',json,broadcast=True)

@socketio.on('mymessage',namespace='/InternReview')
def hanlde_mymessage_Review(json):
    emit('mymessage',json) 