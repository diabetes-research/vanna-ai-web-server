# Vanna.AI-webserver
Web server for chatting with your database

# Setup

## Set your environment variables
```
OPENAI_API_KEY=<Please provide valid OpenAI API key>
MODEL_NAME=<Please specify a model name (e.g., gpt-3.5-turbo)>
MISTRAL_API_KEY = <Please specify API key>
CHROMA_PATH = <Please set chroma db path>
primary_db_path = <Provide primary SQLite database path>
additional_db_path = <Provide secondary SQLite database path>
```

## Create a virtual environment
```
python -m venv <virtualenv_name>
```

## Install dependencies
```
pip install -r requirements.txt
```

## Run the server
```
python app.py
```
