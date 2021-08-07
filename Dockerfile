FROM ubuntu:latest

RUN apt-get update
RUN apt-get install python


WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY train.py /app
COPY BostonData.csv /app


EXPOSE 8000

CMD ["python3","./train.py"]
