FROM python:3.9 
WORKDIR /app
COPY . /app
RUN pip install "distilabel[argilla]"
EXPOSE 80
CMD ["distilabel"]
