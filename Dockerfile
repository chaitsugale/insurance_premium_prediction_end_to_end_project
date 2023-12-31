#step 1 creating base image [linux image]
FROM python:3.11.3

#step 2 : copy -->from my local reportostpry the file should put in the base image--> app floder will get created in container
COPY . /app

#step 3 : working directory
WORKDIR /app

#step 4 : run --> installation of libraries

RUN apt update -y && apt install awscli -y

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 unzip -y && pip install -r requirements.txt

#step 5 : cmd command --> running the command
CMD streamlit run app.py