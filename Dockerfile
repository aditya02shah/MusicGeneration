FROM python:3.10
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install tensorflow==2.12.0
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]