from dotenv import load_dotenv
load_dotenv()

from functools import wraps
from flask import Flask, jsonify, Response, request, redirect, url_for, make_response, render_template_string, render_template
from typing import List, Optional
import os
from cache import MemoryCache
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


vn = MyVanna(config={"model": model, "path": chroma_path})
db_credentials = {
    "host": os.getenv("REMOTE_HOST"),
    "user": os.getenv("REMOTE_UNAME"),
    "password": os.getenv("REMOTE_PASSWD"),
    "database": os.getenv("DB_NAME"),
    "port": os.getenv("PORT"),
}
vn.connect_to_postgres(
    host=db_credentials["host"],
    dbname=db_credentials["database"],
    user=db_credentials["user"],
    password=db_credentials["password"],
    port=db_credentials["port"],
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


app = CustomVannaFlask(vn=vn, cache=cache)
#memory_cache = MemoryCache()
flask_app = app.flask_app
CORS(flask_app)

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
            id:
              type: string
      400:
        description: Invalid input.
    """
    sql = None
    result = None
    df_html = None
    id = None
    if request.method == "OPTIONS":
        return '', 200
    if request.method == "POST":
        # Retrieve the question from the request body
        data = request.get_json()
        question = data.get('question')

        if question:
            try:
                id = cache.generate_id(question=question)
                sql = vn.generate_sql(question, allow_llm_to_see_data=True)
                cache.set(id=id, field="question", value=question)
                cache.set(id=id, field="sql", value=sql)
                # Run SQL query and fetch result
                result = vn.run_sql(sql)
                # Assuming the result is a DataFrame
                df = pd.DataFrame(result)
                # Convert the DataFrame to HTML
                df_html = df.to_html(classes="table table-bordered", index=False)


            except Exception as e:
                result = "Sorry, I couldn't quite understand your question. Could you please rephrase or provide more details?"
                return jsonify({'error': result})

    return jsonify({'sql': sql, 'df_html': df_html, 'id': id})

if __name__ == '__main__':
    host = '0.0.0.0'  # Replace with your desired IP address
    port = 5000         # Replace with your desired port number

    http_server = WSGIServer((host, port), flask_app)
    http_server.serve_forever()
