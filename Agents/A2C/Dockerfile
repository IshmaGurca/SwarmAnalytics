FROM python:3.5
RUN mkdir -p /usr/src/app
RUN mkdir -p /usr/src/app/log
WORKDIR /usr/src/app
COPY requirements.txt /usr/src/app/
RUN pip install --no-cache-dir -r requirements.txt
#ON DEBUG RUN PIP LIB FROM VOLUME
#RUN pip install -e /shared/MessageUtilities
COPY run.py /usr/src/app
COPY ./baselines /usr/src/app/baselines
#CMD ["tensorboard","--logdir=/usr/src/app/log"]
#CMD [ "python","-u", "/usr/src/app/run.py" ]

COPY cmd.sh /usr/src/app/cmd.sh 
CMD ["bash","/usr/src/app/cmd.sh"]
#CMD [ "python","-u", "/usr/src/app/run.py" ]