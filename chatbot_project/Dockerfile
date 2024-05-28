#select base image
FROM python:3.12.3
#create directory
WORKDIR /chatbot_docker
#copy requirements.txt in docker image
COPY chatbot_project/requirements.txt .
#to install requirements..txt
RUN pip install -r requirements.txt
#copy installed dependencies into current directory
COPY . .
#to expose application port
EXPOSE 8000
#to run application in docker container
CMD ["python","manage.py","runserver","0.0.0.0:8004"]
