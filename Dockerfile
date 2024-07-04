FROM python:3.9

WORKDIR /usr/src/inference_api

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY best.pt ./
COPY . .

CMD ["python", "inference_api.py"]
