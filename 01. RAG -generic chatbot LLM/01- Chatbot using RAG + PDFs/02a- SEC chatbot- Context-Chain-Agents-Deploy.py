# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # 2/ Creating the chatbot with Retrieval Augmented Generation (RAG) and DBRX Instruct
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-managed-flow-2.png?raw=true" style="float: right; margin-left: 10px"  width="900px;">
# MAGIC
# MAGIC Our Vector Search Index is now ready!
# MAGIC
# MAGIC Let's now create and deploy a new Model Serving Endpoint to perform RAG.
# MAGIC
# MAGIC The flow will be the following:
# MAGIC
# MAGIC - A user asks a question
# MAGIC - The question is sent to our serverless Chatbot RAG endpoint
# MAGIC - The endpoint compute the embeddings and searches for docs similar to the question, leveraging the Vector Search Index
# MAGIC - The endpoint creates a prompt enriched with the doc
# MAGIC - The prompt is sent to the DBRX Instruct Foundation Model Serving Endpoint
# MAGIC - We display the output to our users!
# MAGIC
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=%2F01-quickstart%2F02-Deploy-RAG-Chatbot-Model&demo_name=llm-rag-chatbot&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fllm-rag-chatbot%2F01-quickstart%2F02-Deploy-RAG-Chatbot-Model&version=1">

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## !! STOP !! Read before proceeding
# MAGIC
# MAGIC Please run the [**01. PDF Ingestion Data Preparation**]($./01. PDF Ingestion Data Preparation) before running this notebook. Ensure that you are using the same `Catalog`, `Database Name`, `Vector Search Endpoint Name` as Notebook 01. The variables Dataset (`Finance- SECFilings`) and Reset Data (`False`) have been preset already and should not be changed. They are still provided to maintain consistency across the notebooks.

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
dbutils.widgets.dropdown("reset_all_data", "true", ["true"], "Reset Data")
dbutils.widgets.dropdown("dataset", "Finance-SecFilings", ["Finance-SecFilings"],"Dataset")

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

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Exploring Langchain capabilities
# MAGIC
# MAGIC Let's start with the basics and send a query to a Databricks Foundation Model using LangChain.
# MAGIC
# MAGIC ### Langchain retriever
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-managed-model-1.png?raw=true" style="float: right" width="500px">
# MAGIC
# MAGIC Let's start by building our Langchain retriever. 
# MAGIC
# MAGIC It will be in charge of:
# MAGIC
# MAGIC * Creating the input question (our Managed Vector Search Index will compute the embeddings for us)
# MAGIC * Calling the vector search index to find similar documents to augment the prompt with 
# MAGIC
# MAGIC Databricks Langchain wrapper makes it easy to do in one step, handling all the underlying logic and API call for you.
# MAGIC

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
print(chain.invoke({"question": "What does Apple do"}))

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

if "Fin" in dataset:
  index_name = f"{catalog}.{dbName}.sec_filings_managed_vs_index"

# COMMAND ----------

# DBTITLE 1,Fetching the Similar documents from Vector Search
from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnableLambda
from operator import itemgetter
import os

host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
os.environ["DATABRICKS_TOKEN"] = dbutils.secrets.get("labs", "sp_token")


embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

def get_retriever(persist_dir: str = None):
    os.environ["DATABRICKS_HOST"] = host
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=host, personal_access_token=os.environ["DATABRICKS_TOKEN"])
    #vsc = VectorSearchClient()
    vs_index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=index_name
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="content_chunk", embedding=embedding_model, columns=["path"]
    )
    return vectorstore.as_retriever(search_kwargs={'k': 4})


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ####!! Important !!
# MAGIC
# MAGIC As a security best practice for production scenarios, Databricks recommends that you use [machine-to-machine OAuth tokens](https://docs.databricks.com/en/dev-tools/auth/oauth-m2m.html) for authentication during production.
# MAGIC
# MAGIC For testing and development, Databricks recommends using a personal access token belonging to [service principals](https://docs.databricks.com/en/admin/users-groups/service-principals.html) instead of workspace users. To create tokens for service principals, see Manage tokens for a service principal.

# COMMAND ----------

# DBTITLE 1,This step is neded for model serving so that the SP has access to read the data
from databricks.sdk import WorkspaceClient

host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
w = WorkspaceClient(token=os.environ["DATABRICKS_TOKEN"], host=host) # make sure you are using the same tokn to register the model.
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

retriever = get_retriever()
query = "what does Apple do?"
similar_docs = retriever.get_relevant_documents(query)
print(f"Number of Relevant docs: {len(similar_docs)}")

# COMMAND ----------

similar_docs

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC We will use a custom LangChain template for our assistant to give a proper answer.
# MAGIC
# MAGIC Make sure you take some time to try different templates and adjust your assistant tone and personality for your requirement.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Let's combine
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatDatabricks

TEMPLATE = """You are an assistant for financial analysis users. You are answering questions related to SEC filings. If the question is not related to SEC filings, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible.
Use the following pieces of context to answer the question at the end:
{context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=get_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

# COMMAND ----------


question = {"query": query}
answer = chain.run(question)
print(answer)

# COMMAND ----------

# DBTITLE 1,Let's also return the source Document for verification purposes

chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=get_retriever(),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)
    

# COMMAND ----------

question = {"query": query}
answer = chain.invoke(question)
print(answer)

# COMMAND ----------

answer.keys()

# COMMAND ----------

answer["result"]

# COMMAND ----------

answer["source_documents"]

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **You can register this as-is for a simple chatbot or can add layers of complex chains and agents to accomplish context awareness, filtering or even calling agents that call python or sql code.** 

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Register the chatbot model to Unity Catalog

# COMMAND ----------

import cloudpickle
import langchain
from mlflow.models import infer_signature

mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog}.{dbName}.sec_chatbot_model"

with mlflow.start_run(run_name="GenAI_WS_chatbot_rag") as run:
    model_info = mlflow.langchain.log_model(
        chain,
        loader_fn= get_retriever,
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch"
        ],
        input_example=question
    )

# COMMAND ----------

# MAGIC %md Let's try loading our model

# COMMAND ----------

model = mlflow.langchain.load_model(model_info.model_uri)
query = "what was the apple's revenue from streaming media?"
model.invoke(query)

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
serving_endpoint_name = f"GenAI_workshop_{catalog}_{dbName}"[:63]
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
question = "How much Apple revenue increase or decrease? provide bullets"

answer = w.serving_endpoints.query(serving_endpoint_name, inputs=[{"query": question}])
print(answer.predictions[0]["result"])

# COMMAND ----------

print(answer.predictions[0]["source_documents"])

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
# MAGIC As a self- work, highly recommend you to explore the [demo on Databricks page](https://www.databricks.com/resources/demos/tutorials/data-science-and-ai/lakehouse-ai-deploy-your-llm-chatbot?itm_data=demo_center). This includes a lot of examples.
# MAGIC
# MAGIC
