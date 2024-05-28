#select base image
FROM python:3.12.3
#create directory
WORKDIR /chatbot_docker
#copy requirements.txt in docker image
# Copy the current directory contents into the container at /chatbot_docker
#COPY . /chatbot_docker
COPY chatbot_project/requirements.txt .
#to install requirements..txt
RUN pip install -r requirements.txt
#copy installed dependencies into current directory
COPY . .
COPY chatbot_project/ /chatbot_docker/
#to expose application port
EXPOSE 8004
#to run application in docker container
CMD ["python","manage.py","runserver","0.0.0.0:8004"]
