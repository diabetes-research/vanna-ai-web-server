from dotenv import load_dotenv
load_dotenv()

from functools import wraps
from flask import Flask, jsonify, Response, request, redirect, url_for, make_response, render_template_string, render_template
from typing import List, Optional
import os
import sqlite3
from cache import MemoryCache
import openai
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
from gevent.pywsgi import WSGIServer
from vanna.flask import VannaFlaskApp, Cache
from vanna.base.base import VannaBase
from vanna.flask.auth import AuthInterface , NoAuth
from urllib.parse import urlparse
import pandas as pd
import json
import requests
from vanna.mistral import Mistral
from flasgger import Swagger
from flask_cors import CORS

flask_app = Flask(__name__, static_folder='assets')
CORS(flask_app)
swagger = Swagger(flask_app)


# SETUP
cache = MemoryCache()
# from vanna.local import LocalContext_OpenAI
# vn = LocalContext_OpenAI()

# Set your OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")

model = os.getenv("MODEL_NAME")
schema_names = os.getenv("SCHEMA_NAMES")
chroma_path = os.getenv("CHROMA_PATH")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


class MyVanna(ChromaDB_VectorStore, Mistral):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        Mistral.__init__(self, config={'api_key': MISTRAL_API_KEY, 'model': model})

    def connect_to_sqlite_multi(self, primary_url: str, additional_urls: list = None):
        """
        Connect to a primary SQLite database and optionally attach multiple additional databases.

        Args:
            primary_url (str): The URL or local path of the primary database to connect to.
            additional_urls (list, optional): List of URLs or local paths for additional databases to attach.

        Returns:
            None
        """

        # Helper function to handle both URLs and local paths
        def get_database_path(url):
            if os.path.exists(url):
                # If it's a local file path, just return it
                return url
            else:
                # If it's a URL, download it
                path = os.path.basename(urlparse(url).path)
                if not os.path.exists(path):
                    response = requests.get(url)
                    response.raise_for_status()  # Check that the request was successful
                    with open(path, "wb") as f:
                        f.write(response.content)
                return path

        # Helper function to extract alias from the path based on 'DRH'
        def extract_alias(url):
            # Extract the part after 'DRH' and before the file extension
            basename = os.path.basename(url)
            alias = basename.split('DRH.')[1].split('.')[0]
            return alias

        # Connect to the primary database
        primary_path = get_database_path(primary_url)
        conn = sqlite3.connect(primary_path)

        # Attach additional databases, if provided
        if additional_urls:
            for url in additional_urls:
                additional_path = get_database_path(url)
                alias = extract_alias(url)  # Generate alias based on content after 'DRH'
                conn.execute(f"ATTACH DATABASE '{additional_path}' AS {alias}")

        # Define run_sql function for executing queries
        def run_sql_sqlite(sql: str):
            return pd.read_sql_query(sql, conn)

        # Set the run_sql function to be used
        self.run_sql = run_sql_sqlite
        self.run_sql_is_set = True


vn = MyVanna(config={"model": model, "path": chroma_path})
vn.connect_to_sqlite_multi(
        primary_url=os.getenv('primary_db_path'),
        additional_urls=os.getenv('additional_db_path').split(',')
    )


def get_schema_name(env_var: str) -> Optional[str]:
    """Fetch the schema name from the environment variable."""
    schema_name: Optional[str] = os.getenv(env_var)

    if schema_name:
        print(f"Schema Name from Environment Variable: {schema_name}")
        return schema_name
    else:
        print(f"No schema name found in environment variable: {env_var}")
        return None


def get_default_schema_name(schema_name: Optional[str]) -> Optional[str]:
    """Return the schema name if available."""
    return schema_name


# Fetch the schema name from the environment variable
schema_name: Optional[str] = get_schema_name("SCHEMA_NAMES")

# Get the default schema name (since it's a string, it will just return it)
default_schema_name: Optional[str] = get_default_schema_name(schema_name)

# Log the schema name and prepare the SQL query
if default_schema_name:
    print(f"Schema Name: {default_schema_name}")
    sql_query: str = f"SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = '{default_schema_name}'"

    # Train the model with the schema name in the SQL query
    vn.train(
        question=f"What are the table and column names in the '{default_schema_name}' schema?",
        sql=sql_query,
    )
else:
    vn.train(
        question="What is the result of a basic SQL query that returns a constant value?",
        sql="SELECT 1;",
    )
    print("No valid schema name found, setting default initial training data.")


class CustomVannaFlask(VannaFlaskApp):
    def __init__(
            self,
            vn: VannaBase,
            cache: Cache = MemoryCache(),
            auth: AuthInterface = NoAuth(),
            debug=False,
            allow_llm_to_see_data=True,
            logo="https://img.vanna.ai/vanna-flask.svg",
            title="English To SQL",
            subtitle="",
            show_training_data=True,
            suggested_questions=True,
            sql=True,
            table=True,
            csv_download=True,
            chart=True,
            redraw_chart=True,
            auto_fix_sql=True,
            ask_results_correct=True,
            followup_questions=True,
            summarization=False,
            function_generation=False,
            index_html_path=None,
            assets_folder=None,
    ):
        # Initialize the parent class VannaFlaskApp
        super().__init__(
            vn,
            cache,
            auth,
            debug,
            allow_llm_to_see_data,
            logo,
            title,
            subtitle,
            show_training_data,
            suggested_questions,
            sql,
            table,
            csv_download,
            chart,
            redraw_chart,
            auto_fix_sql,
            ask_results_correct,
            followup_questions,
            summarization,
            function_generation,
            index_html_path,
            assets_folder,
        )
        self.override_routes()

    def override_routes(self):
        """Override routes with custom implementations."""
        self.flask_app.view_functions.pop("generate_followup_questions", None)

        @self.flask_app.route("/api/v0/generate_followup_questions", methods=["GET"])
        @self.requires_auth
        @self.requires_cache(["df", "question", "sql"])
        def generate_followup_questions(user: any, id: str, df, question, sql):
            """
            Overrides Generate followup questions function by limiting the dataframe rows
            ---
            parameters:
              - name: user
                in: query
              - name: id
                in: query|body
                type: string
                required: true
            responses:
              200:
                schema:
                  type: object
                  properties:
                    type:
                      type: string
                      default: question_list
                    questions:
                      type: array
                      items:
                        type: string
                    header:
                      type: string
            """
            if self.allow_llm_to_see_data:
                followup_questions = self.vn.generate_followup_questions(
                    question=question, sql=sql, df=df.head(100)
                )

                if followup_questions is not None and len(followup_questions) > 5:
                    followup_questions = followup_questions[:5]

                self.cache.set(id=id, field="followup_questions", value=followup_questions)

                return jsonify(
                    {
                        "type": "question_list",
                        "id": id,
                        "questions": followup_questions,
                        "header": "Here are some potential followup questions:",
                    }
                )
            else:
                self.cache.set(id=id, field="followup_questions", value=[])
                return jsonify(
                    {
                        "type": "question_list",
                        "id": id,
                        "questions": [],
                        "header": "Followup Questions can be enabled if you set allow_llm_to_see_data=True",
                    }
                )


app = CustomVannaFlask(vn=vn, cache=MemoryCache())
memory_cache = MemoryCache()
flask_app = app.flask_app


@flask_app.route("/api/v0/ask_question_and_run_query", methods=["POST","OPTIONS"])
def ask_question_and_run_query():
    """
    Ask a question and run the SQL query generated based on the question.
    ---
    parameters:
      - name: question
        in: body
        type: string
        required: true
        description: The question to generate the SQL query from.
    responses:
      200:
        description: A JSON object containing the SQL query and HTML representation of the result.
        schema:
          type: object
          properties:
            sql:
              type: string
              description: The generated SQL query.
            df_html:
              type: string
              description: The HTML representation of the query result.
      400:
        description: Invalid input.
    """
    sql = None
    result = None
    df_html = None
    if request.method == "OPTIONS":
        return '', 200
    if request.method == "POST":
        # Retrieve the question from the request body
        data = request.get_json()
        question = data.get('question')

        if question:
            try:
                sql = vn.generate_sql(question, allow_llm_to_see_data=True)
                # Run SQL query and fetch result
                result = vn.run_sql(sql)
                # Assuming the result is a DataFrame
                df = pd.DataFrame(result)
                # Convert the DataFrame to HTML
                df_html = df.to_html(classes="table table-bordered", index=False)

            except Exception as e:
                result = f"An error occurred: {e}"
                print(result)
                return jsonify({'error': result}), 400  # Return error with status code

    return jsonify({'sql': sql, 'df_html': df_html})


if __name__ == '__main__':
    host = '0.0.0.0'  # Replace with your desired IP address
    port = 5000         # Replace with your desired port number

    http_server = WSGIServer((host, port), app.flask_app)
    http_server.serve_forever()