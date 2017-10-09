FROM python:3.5
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
COPY requirements.txt /usr/src/app/
RUN pip install --no-cache-dir -r requirements.txt
COPY runserver.py /usr/src/app
COPY ./Server /usr/src/app/Server
CMD [ "python", "-u", "runserver.py" ]