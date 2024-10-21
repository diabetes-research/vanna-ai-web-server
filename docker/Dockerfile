From python:3.10.12
ARG GITHUB_TOKEN
RUN apt-get update && \
    apt-get install -y git
RUN git clone https://$GITHUB_TOKEN@github.com/diabetes-research/vanna-ai.git /opt/diabetes-research-ai
WORKDIR /opt/diabetes-research-ai

# Upgrade pip if necessary
RUN pip install --upgrade pip

COPY .env .
RUN mkdir /opt/data
RUN pip install -r requirements.txt
RUN pip install chromadb vanna[postgres] gevent
CMD ["python", "app.py"]
