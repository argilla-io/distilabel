FROM python:3.11-slim 
WORKDIR /app
COPY . /app
RUN pip install "distilabel[argilla]"
EXPOSE 80
CMD ["distilabel"]
