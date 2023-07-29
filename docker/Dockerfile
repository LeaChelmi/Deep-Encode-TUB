FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


COPY get_static_ffmpeg.sh ./
RUN chmod u+x get_static_ffmpeg.sh
RUN ./get_static_ffmpeg.sh

#RUN PATH="$PATH:/usr/src/app/ffmpeg-git-20230721-amd64-static"

RUN ls

RUN mv ffmpeg-git-20230721-amd64-static/ffmpeg ffmpeg-git-20230721-amd64-static/ffprobe /usr/local/bin/

#COPY /usr/src/app/ffmpeg-git-20230721-amd64-static/ffmpeg /usr/local/bin/

RUN ffmpeg -version

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
#RUN apt-get update && apt-get upgrade && apt policy ffmpeg && apt-get install ffmpeg=6.0 libsm6 libxext6  -y
#RUN apt-get update && apt-get install ffmpeg -y



COPY . .

CMD [ "python3", "./main.py" ]