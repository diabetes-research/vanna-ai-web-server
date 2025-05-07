from dotenv import load_dotenv

load_dotenv()

from functools import wraps
from flask import (
    Flask,
    jsonify,
    Response,
    request,
    redirect,
    url_for,
    make_response,
    render_template_string,
    render_template,
)
from typing import List, Optional
import os
from cache import MemoryCache
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
from gevent.pywsgi import WSGIServer
from vanna.flask import VannaFlaskApp, Cache
from vanna.base.base import VannaBase
from vanna.flask.auth import AuthInterface, NoAuth
from urllib.parse import urlparse
import pandas as pd
import json
import requests
from vanna.mistral import Mistral
from flasgger import Swagger
from flask_cors import CORS
import psycopg2
import bcrypt

flask_app = Flask(__name__, static_folder="assets")
CORS(flask_app)
swagger = Swagger(flask_app)


# SETUP
cache = MemoryCache()

model = os.getenv("MODEL_NAME")
schema_names = os.getenv("SCHEMA_NAMES")
chroma_path = os.getenv("CHROMA_PATH")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

db_credentials = {
    "host": os.getenv("REMOTE_HOST"),
    "user": os.getenv("REMOTE_UNAME"),
    "password": os.getenv("REMOTE_PASSWD"),
    "database": os.getenv("DB_NAME"),
    "port": os.getenv("PORT"),
}


def create_db_connection():
    conn = psycopg2.connect(
        dbname=db_credentials["database"],
        user=db_credentials["user"],
        password=db_credentials["password"],
        host=db_credentials["host"],
        port=db_credentials["port"],
    )
    return conn


db_connection = create_db_connection()


def get_auth_db_connection():
    return psycopg2.connect(
        dbname=db_credentials["database"],
        user=os.getenv("AUTH_DB_USER"),
        password=os.getenv("AUTH_DB_PASSWORD"),
        host=db_credentials["host"],
        port=db_credentials["port"],
    )


class SimplePassword(AuthInterface):
    def __init__(self, db_connection):
        self.db_connection = db_connection

    def get_user(self, flask_request) -> any:
        return flask_request.cookies.get("user")

    def is_logged_in(self, user: any) -> bool:
        return user is not None

    def override_config_for_user(self, user: any, config: dict) -> dict:
        role = request.cookies.get("role")  # Retrieve the role from the cookie

        if role == "vanna_user_admin":
            config["show_training_data"] = True
            config["ask_results_correct"] = True
        elif role == "vanna_user_normal":
            config["show_training_data"] = False
            config["ask_results_correct"] = False
        return config

    def login_form(self) -> str:
        return """
         <div class="p-4 sm:p-7">
        <div class="text-center">
        <h1 class="block text-2xl font-bold text-gray-800 dark:text-white">Sign in</h1>
        <p class="mt-2 text-sm text-gray-600 dark:text-gray-400">

        </p>
        </div>

        <div class="mt-5">

        <!-- Form -->
        <form action="/auth/login" method="POST">
            <div class="grid gap-y-4">
            <!-- Form Group -->
          <div>
            <label for="email" class="block text-sm mb-2 dark:text-white">Email address</label>
            <div class="relative">
              <input type="email" id="email" type="email" name="email" class="py-3 px-4 block w-full border border-gray-200 rounded-lg text-sm focus:border-blue-500 focus:ring-blue-500 disabled:opacity-50 disabled:pointer-events-none dark:bg-slate-900 dark:border-gray-700 dark:text-gray-400 dark:focus:ring-gray-600" required aria-describedby="email-error">
              <div class="hidden absolute inset-y-0 end-0 pointer-events-none pe-3">
                <svg class="size-5 text-red-500" width="16" height="16" fill="currentColor" viewBox="0 0 16 16" aria-hidden="true">
                  <path d="M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0zM8 4a.905.905 0 0 0-.9.995l.35 3.507a.552.552 0 0 0 1.1 0l.35-3.507A.905.905 0 0 0 8 4zm.002 6a1 1 0 1 0 0 2 1 1 0 0 0 0-2z"/>
                </svg>
              </div>
            </div>
            <p class="hidden text-xs text-red-600 mt-2" id="email-error">Please include a valid email address so we can get back to you</p>
          </div>
          <!-- End Form Group -->

          <!-- Form Group -->
          <div>
            <div class="flex justify-between items-center">
              <label for="password" class="block text-sm mb-2 dark:text-white">Password</label>

            </div>
            <div class="relative">
              <input type="password" id="password" name="password" class="py-3 px-4 block w-full border border-gray-200 rounded-lg text-sm focus:border-blue-500 focus:ring-blue-500 disabled:opacity-50 disabled:pointer-events-none dark:bg-slate-900 dark:border-gray-700 dark:text-gray-400 dark:focus:ring-gray-600" required aria-describedby="password-error">
              <div class="hidden absolute inset-y-0 end-0 pointer-events-none pe-3">
                <svg class="size-5 text-red-500" width="16" height="16" fill="currentColor" viewBox="0 0 16 16" aria-hidden="true">
                  <path d="M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0zM8 4a.905.905 0 0 0-.9.995l.35 3.507a.552.552 0 0 0 1.1 0l.35-3.507A.905.905 0 0 0 8 4zm.002 6a1 1 0 1 0 0 2 1 1 0 0 0 0-2z"/>
                </svg>
              </div>
            </div>
            <p class="hidden text-xs text-red-600 mt-2" id="password-error">8+ characters required</p>
          </div>
          <!-- End Form Group -->

          <button type="submit" class="w-full py-3 px-4 inline-flex justify-center items-center gap-x-2 text-sm font-semibold rounded-lg border border-transparent bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 disabled:pointer-events-none">Sign in</button>
        </div>
        </form>
      <!-- Registration Link -->
                <p class="mt-2 text-center text-sm text-gray-600 dark:text-gray-400">
                    <!--Don't have an account? <a href="/register" class="text-blue-600 hover:text-blue-500">Register here</a> -->
                </p>
        </div>
    </div>


         """

    def login_handler(self, flask_request) -> str:
        email = flask_request.form["email"]
        password = flask_request.form["password"]

        auth_conn = get_auth_db_connection()
        auth_cursor = auth_conn.cursor()

        try:
            auth_cursor.execute(
                """
                SELECT username, password_hash, user_role
                FROM drh_stateless_vanna.vanna_user_profile_view
                WHERE username = %s
            """,
                (email,),
            )

            user = auth_cursor.fetchone()

            if user:
                stored_username = user[0]
                stored_password_hash = user[1]
                user_role = user[2]

                if bcrypt.checkpw(password.encode("utf-8"), stored_password_hash.encode("utf-8")):
                    response = make_response("Logged in as " + stored_username)
                    response.set_cookie("user", stored_username)
                    response.set_cookie("role", user_role)
                    response.headers["Location"] = "/"
                    response.status_code = 302
                    return response
                else:
                    return "Invalid password"
            else:
                return "Login failed: User not found"

        finally:
            auth_cursor.close()
            auth_conn.close()


    def callback_handler(self, flask_request) -> str:
        user = flask_request.args["user"]
        response = make_response("Logged in as " + user)
        response.set_cookie("user", user)
        return response

    def logout_handler(self, flask_request) -> str:
        response = make_response(redirect(url_for("login")))
        response.delete_cookie("user")
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

auth = SimplePassword(db_connection)


class MyVanna(ChromaDB_VectorStore, Mistral):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        Mistral.__init__(self, config={"api_key": MISTRAL_API_KEY, "model": model})

    def get_sql_prompt(
        self,
        initial_prompt: str,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        **kwargs,
    ):
        """
        This method is used to generate a prompt for the LLM to generate SQL.

        Args:
            question (str): The question to generate SQL for.
            question_sql_list (list): A list of questions and their corresponding SQL statements.
            ddl_list (list): A list of DDL statements.
            doc_list (list): A list of documentation.

        Returns:
            any: The prompt for the LLM to generate SQL.
        """

        if initial_prompt is None:
            initial_prompt = (
                f"You are a {self.dialect} expert. "
                + "Please help to generate a SQL query to answer the question. Your response should ONLY be based on the given context and follow the response guidelines and format instructions. "
            )

        initial_prompt = self.add_ddl_to_prompt(initial_prompt, ddl_list, max_tokens=self.max_tokens)

        if self.static_documentation != "":
            doc_list.append(self.static_documentation)

        initial_prompt = self.add_documentation_to_prompt(initial_prompt, doc_list, max_tokens=self.max_tokens)

        initial_prompt += (
            "=== Response Guidelines ===\n"
            "1. If the provided context is sufficient, generate a valid SQL query without any explanations.\n"
            "2. If the context is almost sufficient but requires identifying specific string values in a column, generate an intermediate SQL query to retrieve distinct strings in that column. Prepend the query with a comment saying 'intermediate_sql'.\n"
            "3. If the context is insufficient, briefly explain why an SQL query cannot be generated.\n"
            "4. Use only the most relevant table(s) from the application's schema.\n"
            "5. If the same question has been answered before, repeat the previous answer exactly.\n"
            f"6. Ensure that the output SQL is {self.dialect}-compliant and free of syntax errors.\n\n"
            "Note: Do not answer questions that involve:\n"
            "- PostgreSQL internal tables (e.g., pg_user, pg_settings, information_schema, pg_catalog)\n"
            "- user details, passwords, configurations, permissions or roles\n"
            "- Listing or managing users, roles, or permissions\n"
            "- Server settings or infrastructure\n"
            "For such questions, respond with:\n"
            "\"I'm sorry. To maintain a secure environment, I don’t access sensitive or system-level information. I'm here to help with DRH-related queries only.\""
        )

        message_log = [self.system_message(initial_prompt)]

        for example in question_sql_list:
            if example is None:
                print("example is None")
            else:
                if example is not None and "question" in example and "sql" in example:
                    message_log.append(self.user_message(example["question"]))
                    message_log.append(self.assistant_message(example["sql"]))

        message_log.append(self.user_message(question))

        return message_log


vn = MyVanna(config={"model": model, "path": chroma_path})
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
    sql_query: str = (
        f"SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = '{default_schema_name}'"
    )

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
                followup_questions = self.vn.generate_followup_questions(question=question, sql=sql, df=df.head(100))

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

        # overriding /train to add training logs to database
        self.flask_app.view_functions.pop("add_training_data", None)

        @self.flask_app.route("/api/v0/train", methods=["POST"])
        @self.requires_auth
        def add_training_data(user: any):
            """
            Add training data
            ---
            parameters:
              - name: user
                in: query
              - name: question
                in: body
                type: string
              - name: sql
                in: body
                type: string
              - name: ddl
                in: body
                type: string
              - name: documentation
                in: body
                type: string
            responses:
              200:
                schema:
                  type: object
                  properties:
                    id:
                      type: string
            """
            question = request.json.get("question")
            sql = request.json.get("sql")
            ddl = request.json.get("ddl")
            documentation = request.json.get("documentation")

            try:
                id = vn.train(question=question, sql=sql, ddl=ddl, documentation=documentation)
                df = vn.get_training_data()
                question_row = df[df["id"] == id]
                if not question_row.empty:
                    question = question_row.iloc[0]["question"]

                user_name = request.cookies.get("user")

                cursor = db_connection.cursor()
                training_data = ""
                if sql:
                    training_data = sql
                elif ddl:
                    training_data = ddl
                    question = "ddl"
                elif documentation:
                    training_data = documentation
                    question = "documentation"
                else:
                    training_data = "No training data"
                    question = "No question"
                cursor.execute(
                    """ SELECT drh_stateless_vanna.log_vanna_ai_response(
                                                                %s,  -- question
                                                                %s,  -- training_data
                                                                %s,  -- created_by
                                                                %s,  -- query_type
                                                                %s,  -- son_result
                                                                %s   -- results
                                                            )
                                                        """,
                    (question, training_data, user_name, "training", None, None),
                )
                db_connection.commit()
                return jsonify({"id": id})

            except Exception as e:
                print("TRAINING ERROR", e)
                return jsonify({"type": "error", "error": str(e)})


app = CustomVannaFlask(vn=vn, cache=cache, auth=auth)
flask_app = app.flask_app
CORS(flask_app)

@flask_app.route("/login-error")
def login_error():
    message = request.args.get("message", "")
    message_type = request.args.get("message_type", "info")
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Login Error</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <style>
            body {{
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                background-color: #f8f9fa;
            }}
            .error-container {{
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }}
            .error-message {{
                color: {'red' if message_type == 'error' else 'green'};
            }}
        </style>
    </head>
    <body>
        <div class="error-container text-center">
            <h2 class="error-message">{message}</h2>            
            <a href="/login" class="btn btn-primary">Go to Login</a>
        </div>
    </body>
    </html>
    """


@flask_app.route("/api/v0/ask_question_and_run_query", methods=["POST", "OPTIONS"])
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
        return "", 200

    if request.method == "POST":
        # Retrieve the question from the request body
        data = request.get_json()
        question = data.get("question")
        user_role = request.cookies.get("role")
        print("user role", user_role)

        if not question:
            return jsonify({"error": "Question is required"}), 400

        try:
            # Generate a unique ID for the question
            id = cache.generate_id(question=question)

            # Generate SQL query
            sql = vn.generate_sql(question, allow_llm_to_see_data=True)

            # Check if the LLM response contains the restricted message
            restricted_message = (
                "I'm sorry. To maintain a secure environment, I don’t access sensitive or system-level information. "
                "I'm here to help with DRH-related queries only."
            )
            if sql.strip() == restricted_message:
                return jsonify({"error": restricted_message})

            # Cache the question and SQL
            cache.set(id=id, field="question", value=question)
            cache.set(id=id, field="sql", value=sql)

            # Run SQL query and fetch result
            result = vn.run_sql(sql)

            # Assuming the result is a DataFrame
            df = pd.DataFrame(result)
            # Convert the DataFrame to HTML
            df_html = df.to_html(classes="table table-bordered", index=False)

        except Exception as e:
            return jsonify(
                {
                    "error": "Sorry, I couldn't quite understand your question. Could you please rephrase or provide more details?"
                }
            )

    return jsonify({"sql": sql, "df_html": df_html, "id": id})


if __name__ == "__main__":
    host = "0.0.0.0"  # Replace with your desired IP address
    port = 5000  # Replace with your desired port number

    http_server = WSGIServer((host, port), flask_app)
    http_server.serve_forever()
