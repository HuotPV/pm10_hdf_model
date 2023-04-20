FROM python:3.8
VOLUME /app/outputs
VOLUME /app/inputs

COPY ./requirements.txt /app/requirements.txt
WORKDIR /home/pv/Documents/pm10_hdf_data/
COPY /* /app/outputs/
WORKDIR /app

RUN pip install -r requirements.txt
COPY . /app 

CMD ["python3", "/app/main.py"]