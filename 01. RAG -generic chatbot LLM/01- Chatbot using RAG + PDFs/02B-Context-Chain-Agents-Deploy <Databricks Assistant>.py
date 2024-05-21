# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # 2. Building pipelines to include context, history and reading from the Vector Database
# MAGIC
# MAGIC
# MAGIC Our Vector Search Index is now ready!
# MAGIC
# MAGIC **Chaining RAG systems** offers a powerful approach to leveraging different data sources and AI models, enhancing the quality and applicability of generated content in various applications such as customer support automation and medical research assistance.
# MAGIC
# MAGIC Let's now create a more advanced langchain model to perform RAG.
# MAGIC
# MAGIC We will improve our langchain model with the following:
# MAGIC
# MAGIC - Build a complete chain supporting a chat history, using llama 2 input style
# MAGIC - Add a filter to only answer Databricks-related questions
# MAGIC - Compute the embeddings with Databricks BGE models within our chain to query the self-managed Vector Search Index
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=02-Deploy-RAG-Chatbot-Model&demo_name=chatbot-rag-llm&event=VIEW">
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## !! STOP !! Read before proceeding
# MAGIC
# MAGIC Please run the [**01. PDF Ingestion Data Preparation**]($./01. PDF Ingestion Data Preparation) before running this notebook. Ensure that you are using the same `Catalog`, `Database Name`, `Vector Search Endpoint Name` as Notebook 01. The variables Dataset (`Databricks-Documentation`) and Reset Data (`False`) have been preset already and should not be changed. They are still provided to maintain consistency across the notebooks. 

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## What are Chain and agents?
# MAGIC
# MAGIC * **LLM chains** are formed by connecting one or more large language models in a logical sequence. They integrate an LLM with a prompt, allowing for the execution of operations on text or datasets. Each LLM in the chain may specialize in different aspects of language understanding or generation, and the output of one LLM serves as the input to the next in the chain. 
# MAGIC
# MAGIC * **LLM Agents** are tools that enable LLMs to perform various tasks beyond text generation. They act as "tools" for LLMs, allowing them to execute Python code, search for information, query databases, and more. These agents are typically tailored to excel in particular areas of language understanding or generation, such as sentiment analysis, text summarization, question answering, or natural language understanding for specific industries or domains. 
# MAGIC

# COMMAND ----------

# Import required libraries
from pyspark.sql.functions import col
from pyspark.sql.types import BooleanType
from pyspark.sql.functions import lit

# Remove all widgets
dbutils.widgets.removeAll()

# Create the widgets
dbutils.widgets.text("dbName", "", "Database Name")
dbutils.widgets.text("VECTOR_SEARCH_ENDPOINT_NAME", "", "Vector Search Endpoint Name")
dbutils.widgets.text("catalog", "", "Catalog")
# please note- This notebook will not work as expected if you have reset the data, so there is no True option here
dbutils.widgets.dropdown("reset_all_data", "false", ["false"], "Reset Data")
dbutils.widgets.dropdown("dataset", "Databricks-Documentation", ["Databricks-Documentation"],"Dataset")

# Get the widget values
dbName = dbutils.widgets.get("dbName")
VECTOR_SEARCH_ENDPOINT_NAME = dbutils.widgets.get("VECTOR_SEARCH_ENDPOINT_NAME")
catalog = dbutils.widgets.get("catalog")
reset_all_data = dbutils.widgets.get("reset_all_data")
dataset = dbutils.widgets.get("dataset")


# Convert reset_all_data to boolean
reset_all_data = True if reset_all_data.lower() == "true" else False

# Print the widget values for verification
print(f"dbName: {dbName}")
print(f"VECTOR_SEARCH_ENDPOINT_NAME: {VECTOR_SEARCH_ENDPOINT_NAME}")
print(f"catalog: {catalog}")
print(f"reset_all_data: {reset_all_data}")
print(f"Dataset is: {dataset}")
print("\n Please fill any missing or Incorrect Values....")

# COMMAND ----------

# MAGIC %run ./00-Setup $reset_all_data=$reset_all_data $dbName=$dbName $catalog=$catalog $dataset=$dataset $VECTOR_SEARCH_ENDPOINT_NAME=$VECTOR_SEARCH_ENDPOINT_NAME 

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC select current_catalog(), current_database();

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Exploring Langchain capabilities
# MAGIC
# MAGIC Let's start with the basics and send a query to a Databricks Foundation Model using LangChain.
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Simple
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatDatabricks
from langchain.schema.output_parser import StrOutputParser

prompt = PromptTemplate(
  input_variables = ["question"],
  template = "You are an assistant. Do not attempt to answer if you do not know. Give a short answer to this question: {question}"
)
chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 500)

chain = (
  prompt
  | chat_model
  | StrOutputParser()
)
print(chain.invoke({"question": "What are LLMs?"}))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Let's add a filter on top to only answer Databricks-related questions and conversation history to the prompt 
# MAGIC
# MAGIC
# MAGIC **Conversation History:**
# MAGIC
# MAGIC When invoking our chain, we'll pass history as a list, specifying whether each message was sent by a user or the assistant. For example:
# MAGIC
# MAGIC ```
# MAGIC [
# MAGIC   {"role": "user", "content": "What is Apache Spark?"}, 
# MAGIC   {"role": "assistant", "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."}, 
# MAGIC   {"role": "user", "content": "Does it support streaming?"}
# MAGIC ]
# MAGIC ```
# MAGIC
# MAGIC Let's create chain components to transform this input into the inputs passed to `prompt_with_history`.
# MAGIC
# MAGIC
# MAGIC **Filter:**
# MAGIC
# MAGIC We want our chatbot to be profesionnal and only answer questions related to Databricks. Let's create a small chain and add a first classification step. 
# MAGIC
# MAGIC *Note: this is a fairly-naive implementation, another solution could be adding a small classification model based on the question embedding, providing faster classification*

# COMMAND ----------

from langchain.schema.runnable import RunnableLambda
from operator import itemgetter



#The question is the last entry of the history
def extract_question(input):
    return input[-1]["content"]

#The history is everything before the last question
def extract_history(input):
    return input[:-1]



##Filter
is_question_about_databricks_str = """
You are classifying documents to know if this question is related with Databricks in AWS, Azure and GCP, Workspaces, Databricks account and cloud infrastructure setup, Data Science, Data Engineering, Big Data, Datawarehousing, SQL, Python and Scala or something from a very different field. Also answer no if the last part is inappropriate. 

Here are some examples:

Question: Knowing this followup history: What is Databricks?, classify this question: Do you have more details?
Expected Response: Yes

Question: Knowing this followup history: What is Databricks?, classify this question: Write me a song.
Expected Response: No

Only answer with "yes" or "no". 

Knowing this followup history: {chat_history}, classify this question: {question}
"""

#create prompt with context as filter for databricks chat
is_question_about_databricks_prompt = PromptTemplate(
  input_variables= ["chat_history", "question"],
  template = is_question_about_databricks_str
)

#Create chain with prompt
is_about_databricks_chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | is_question_about_databricks_prompt
    | chat_model
    | StrOutputParser()
)

#Returns "Yes" as this is about Databricks: 
print(is_about_databricks_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is Apache Spark?"}, 
        {"role": "assistant", "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."}, 
        {"role": "user", "content": "Does it support streaming?"}
    ]
}))

# COMMAND ----------

#Return "no" as this isn't about Databricks
print(is_about_databricks_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is the meaning of life?"}
    ]
}))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Use LangChain to retrieve documents from the vector store
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-self-managed-model-1.png?raw=true" style="float: right" width="500px">
# MAGIC
# MAGIC Let's add our LangChain retriever. 
# MAGIC
# MAGIC It will be in charge of:
# MAGIC
# MAGIC * Creating the input question embeddings (with Databricks `bge-large-en`)
# MAGIC * Calling the vector search index to find similar documents to augment the prompt with
# MAGIC
# MAGIC Databricks LangChain wrapper makes it easy to do in one step, handling all the underlying logic and API call for you.

# COMMAND ----------

# DBTITLE 1,Grant Service Principal access to your database, index and model
from databricks.sdk import WorkspaceClient

host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
w = WorkspaceClient(token=dbutils.secrets.get("labs", "sp_token"), host=host)
sp_id = w.current_user.me().emails[0].value
print(f"Service Principal ID: {sp_id}")

##catalog permissions
spark.sql(f'GRANT USAGE ON CATALOG {catalog} TO `{sp_id}`')

# Enable Service Principal (SP) to use the database, select from table and execute model
spark.sql(f"GRANT USE SCHEMA ON DATABASE {dbName} TO `{sp_id}`")
spark.sql(f"GRANT EXECUTE ON DATABASE {dbName} TO `{sp_id}`")
spark.sql(f"GRANT SELECT ON DATABASE {dbName} TO `{sp_id}`")
# If we want to enable inference table for the endpoint we have to give SP permission to create a table in db and modify that table.
spark.sql(f"GRANT CREATE ON DATABASE {dbName} TO `{sp_id}`")
spark.sql(f"GRANT MODIFY ON DATABASE {dbName} TO `{sp_id}`")

print("permissions set")

# COMMAND ----------

index_name=f"{catalog}.{dbName}.databricks_documentation_managed_vs_index"

host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

# #Let's make sure the secret is properly setup and can access our vector search index. Check the quick-start demo for more guidance
# test_demo_permissions(host, secret_scope="dbdemos", secret_key="rag_sp_token", vs_endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, catalog = catalog, db = dbName, index_name=index_name, embedding_endpoint_name="databricks-bge-large-en", managed_embeddings = False)

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnableLambda
from operator import itemgetter
import os

os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("labs", "token")

embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

def get_retriever(persist_dir: str = None):
    os.environ["DATABRICKS_HOST"] = host
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=host, personal_access_token=os.environ["DATABRICKS_TOKEN"])
    vs_index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=index_name
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="content_chunk", embedding=embedding_model, columns=["path"]
    )
    return vectorstore.as_retriever(search_kwargs={'k': 4})

retriever = get_retriever()

retrieve_document_chain = (
    itemgetter("messages") 
    | RunnableLambda(extract_question)
    | retriever
)
print(retrieve_document_chain.invoke({"messages": [{"role": "user", "content": "What is Apache Spark?"}]}))

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnableLambda
from operator import itemgetter

os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("labs", "sp_token")

embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

def get_retriever(persist_dir: str = None):
    os.environ["DATABRICKS_HOST"] = host
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=host, personal_access_token=os.environ["DATABRICKS_TOKEN"])
    vs_index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=index_name
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="content_chunk", embedding=embedding_model, columns=["path"]
    )
    return vectorstore.as_retriever(search_kwargs={'k': 4})

retriever = get_retriever()

retrieve_document_chain = (
    itemgetter("messages") 
    | RunnableLambda(extract_question)
    | retriever
)
print(retrieve_document_chain.invoke({"messages": [{"role": "user", "content": "What is Apache Spark?"}]}))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Improve document search using LLM to generate a better sentence for the vector store, based on the chat history
# MAGIC
# MAGIC We need to retrieve documents related the the last question but also the history.
# MAGIC
# MAGIC One solution is to add a step for our LLM to summarize the history and the last question, making it a better fit for our vector search query. Let's do that as a new step in our chain:

# COMMAND ----------

from langchain.schema.runnable import RunnableBranch

generate_query_to_retrieve_context_template = """
Based on the chat history below, we want you to generate a query for an external data source to retrieve relevant documents so that we can better answer the question. The query should be in natual language. The external data source uses similarity search to search for relevant documents in a vector space. So the query should be similar to the relevant documents semantically. Answer with only the query. Do not add explanation.

Chat history: {chat_history}

Question: {question}
"""

generate_query_to_retrieve_context_prompt = PromptTemplate(
  input_variables= ["chat_history", "question"],
  template = generate_query_to_retrieve_context_template
)

generate_query_to_retrieve_context_chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | RunnableBranch(  #Augment query only when there is a chat history
      (lambda x: x["chat_history"], generate_query_to_retrieve_context_prompt | chat_model | StrOutputParser()),
      (lambda x: not x["chat_history"], RunnableLambda(lambda x: x["question"])),
      RunnableLambda(lambda x: x["question"])
    )
)

#Let's try it
output = generate_query_to_retrieve_context_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is Apache Spark?"}
    ]
})
print(f"Test retriever query without history: {output}")

output = generate_query_to_retrieve_context_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is Apache Spark?"}, 
        {"role": "assistant", "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."}, 
        {"role": "user", "content": "Does it support streaming?"}
    ]
})
print(f"Test retriever question, summarized with history: {output}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Let's put it together
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-self-managed-model-2.png?raw=true" style="float: right" width="600px">
# MAGIC
# MAGIC
# MAGIC Let's now merge the retriever and the full LangChain chain.
# MAGIC
# MAGIC We will use a custom LangChain template for our assistant to give a proper answer.
# MAGIC
# MAGIC Make sure you take some time to try different templates and adjust your assistant tone and personality for your requirement.
# MAGIC
# MAGIC

# COMMAND ----------

from langchain.schema.runnable import RunnableBranch, RunnableParallel, RunnablePassthrough

question_with_history_and_context_str = """
You are a trustful assistant for Databricks users. You are answering python, coding, SQL, data engineering, spark, data science, AI, ML, Datawarehouse, platform, API or infrastructure, Cloud administration question related to Databricks. If you do not know the answer to a question, you truthfully say you do not know. Read the discussion to get the context of the previous conversation. In the chat discussion, you are referred to as "system". The user is referred to as "user".

Discussion: {chat_history}

Here's some context which might or might not help you answer: {context}

Answer straight, do not repeat the question, do not start with something like: the answer to the question, do not add "AI" in front of your answer, do not say: here is the answer, do not mention the context or the question.

Based on this history and context, answer this question: {question}
"""

question_with_history_and_context_prompt = PromptTemplate(
  input_variables= ["chat_history", "context", "question"],
  template = question_with_history_and_context_str
)

def format_context(docs):
    return "\n\n".join([d.page_content for d in docs])

def extract_source_urls(docs):
    return [d.metadata["path"] for d in docs]

relevant_question_chain = (
  RunnablePassthrough() |
  {
    "relevant_docs": generate_query_to_retrieve_context_prompt | chat_model | StrOutputParser() | retriever,
    "chat_history": itemgetter("chat_history"), 
    "question": itemgetter("question")
  }
  |
  {
    "context": itemgetter("relevant_docs") | RunnableLambda(format_context),
    "sources": itemgetter("relevant_docs") | RunnableLambda(extract_source_urls),
    "chat_history": itemgetter("chat_history"), 
    "question": itemgetter("question")
  }
  |
  {
    "prompt": question_with_history_and_context_prompt,
    "sources": itemgetter("sources")
  }
  |
  {
    "result": itemgetter("prompt") | chat_model | StrOutputParser(),
    "sources": itemgetter("sources")
  }
)

irrelevant_question_chain = (
  RunnableLambda(lambda x: {"result": 'I cannot answer questions that are not about Databricks.', "sources": []})
)

branch_node = RunnableBranch(
  (lambda x: "yes" in x["question_is_relevant"].lower(), relevant_question_chain),
  (lambda x: "no" in x["question_is_relevant"].lower(), irrelevant_question_chain),
  irrelevant_question_chain
)

full_chain = (
  {
    "question_is_relevant": is_about_databricks_chain,
    "question": itemgetter("messages") | RunnableLambda(extract_question),
    "chat_history": itemgetter("messages") | RunnableLambda(extract_history),    
  }
  | branch_node
)

# COMMAND ----------

# MAGIC %md 
# MAGIC Let's try our full chain:

# COMMAND ----------

# DBTITLE 1,Asking an out-of-scope question
import json
non_relevant_dialog = {
    "messages": [
        {"role": "user", "content": "What is Apache Spark?"}, 
        {"role": "assistant", "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."}, 
        {"role": "user", "content": "Why is the sky blue?"}
    ]
}
print(f'Testing with a non relevant question...')
response = full_chain.invoke(non_relevant_dialog)
display_chat(non_relevant_dialog["messages"], response)

# COMMAND ----------

# DBTITLE 1,Asking a relevant question
dialog = {
    "messages": [
        {"role": "user", "content": "What is Apache Spark?"}, 
        {"role": "assistant", "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."}, 
        {"role": "user", "content": "Does it support streaming?"}
    ]
}
print(f'Testing with relevant history and question...')
response = full_chain.invoke(dialog)
display_chat(dialog["messages"], response)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Register the chatbot model to Unity Catalog

# COMMAND ----------

import cloudpickle
import langchain
from mlflow.models import infer_signature

mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog}.{dbName}.GenAI_workshop_databricks_doc"

with mlflow.start_run(run_name="dbdemos_chatbot_rag") as run:
    #Get our model signature from input/output
    input_df = pd.DataFrame({"messages": [dialog]})
    output = full_chain.invoke(dialog)
    signature = infer_signature(input_df, output)

    model_info = mlflow.langchain.log_model(
        full_chain,
        loader_fn=get_retriever,  # Load the retriever with DATABRICKS_TOKEN env as secret (for authentication).
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
            "pydantic==2.5.2 --no-binary pydantic",
            "cloudpickle=="+ cloudpickle.__version__
        ],
        input_example=input_df,
        signature=signature
    )

# COMMAND ----------

# MAGIC %md Let's try loading our model

# COMMAND ----------

model = mlflow.langchain.load_model(model_info.model_uri)
model.invoke(dialog)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # Deploying our Chat Model and enabling Online Evaluation Monitoring
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-eval-online-2-0.png?raw=true" style="float: right" width="900px">
# MAGIC
# MAGIC
# MAGIC
# MAGIC Let's now deploy our model as an endpoint to be able to send real-time queries.
# MAGIC
# MAGIC Once our model is live, we will need to monitor its behavior to detect potential anomaly and drift over time. 
# MAGIC
# MAGIC We won't be able to measure correctness as we don't have a ground truth, but we can track model perplexity and other metrics like profesionalism over time.
# MAGIC
# MAGIC This can easily be done by turning on your Model Endpoint Inference table, automatically saving every query input and output as one of your Delta Lake tables.

# COMMAND ----------

import urllib
import json
import mlflow

# Create or update serving endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput

mlflow.set_registry_uri('databricks-uc')
client = MlflowClient()
model_name = f"{catalog}.{dbName}.GenAI_workshop_databricks_doc"
serving_endpoint_name = f"gen_ai_ws_{catalog}_{dbName}"[:63]
latest_model_version = get_latest_model_version(model_name)

w = WorkspaceClient()

serving_client = EndpointApiClient()
# Start the endpoint using the REST API (you can do it using the UI directly)
auto_capture_config = {
    "catalog_name": catalog,
    "schema_name": dbName,
    "table_name_prefix": serving_endpoint_name
    }
environment_vars={"DATABRICKS_TOKEN": "{{secrets/labs/sp_token}}"}
serving_client.create_endpoint_if_not_exists(serving_endpoint_name, model_name=model_name, model_version = latest_model_version, workload_size="Small", scale_to_zero_enabled=True, wait_start = True, auto_capture_config=auto_capture_config, environment_vars=environment_vars)

# COMMAND ----------

displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')

# COMMAND ----------

# DBTITLE 1,Let's try to send a query to our chatbot
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import DataframeSplitInput

df_split = DataframeSplitInput(columns=["messages"],
                               data=[[ {"messages": [{"role": "user", "content": "What is Apache Spark?"}, 
                                                     {"role": "assistant", "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."}, 
                                                     {"role": "user", "content": "Does it support streaming?"}
                                                    ]}]])
w = WorkspaceClient()
w.serving_endpoints.query(serving_endpoint_name, dataframe_split=df_split)

# COMMAND ----------

display_gradio_app("databricks-demos-chatbot")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC We've seen how we can improve our chatbot, adding more advanced capabilities to handle a chat history.
# MAGIC
# MAGIC As you add capabilities to your model and tune the prompt, it will get harder to evaluate your model performance in a repeatable way.
# MAGIC
# MAGIC Your new prompt might work well for what you tried to fixed, but could also have impact on other questions.
# MAGIC
# MAGIC
# MAGIC ## Next Steps:
# MAGIC As a self- work, highly recommend you to explore the next notebooks: set up Inferencing and monitoring using Databricks
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Online LLM evaluation with Databricks Monitoring
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-eval-online-2-2.png?raw=true" style="float: right" width="900px">
# MAGIC
# MAGIC Let's now analyze and monitor our model.
# MAGIC
# MAGIC
# MAGIC Here are the required steps:
# MAGIC
# MAGIC - Make sure the Inference table is enabled (it was automatically setup in the previous cell)
# MAGIC - Consume all the Inference table payload, and measure the model answer metrics (perplexity, complexity etc)
# MAGIC - Save the result in your metric table. This can first be used to plot the metrics over time
# MAGIC - Leverage Databricks Monitoring to analyze the metric evolution over time
