# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC # Scaling your business with a GenAI
# MAGIC
# MAGIC LLMs are disrupting the way we interact with information, from internal knowledge bases to external, customer-facing documentation or support.
# MAGIC
# MAGIC One of the most deployed Gen-AI solutions currently is a chatbot. It can be used for several tasks such as answering your questions, retrieving information quickly or even co-piloting.
# MAGIC  
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/moisaic-logo.png?raw=true" width="100px" style="float: right" />
# MAGIC
# MAGIC While ChatGPT democratized LLM-based chatbots for consumer use, companies need to deploy personalized models that answer their needs:
# MAGIC
# MAGIC - Privacy requirements on sensitive information
# MAGIC - Preventing hallucination
# MAGIC - Specialized content, not available on the Internet
# MAGIC - Specific behavior for customer tasks
# MAGIC - Control over speed and cost
# MAGIC - Deploy models on private infrastructure for security reasons
# MAGIC
# MAGIC ## Introducing Databricks AI
# MAGIC
# MAGIC To solve these challenges, custom knowledge bases and models need to be deployed. However, doing so at scale isn't simple and requires:
# MAGIC
# MAGIC - Ingesting and transforming massive amounts of data 
# MAGIC - Ensuring privacy and security across your data pipeline
# MAGIC - Deploying systems such as Vector Search Index 
# MAGIC - Having access to GPUs and deploying efficient LLMs for inference serving
# MAGIC - Training and deploying custom models
# MAGIC
# MAGIC This is where the Databricks  AI comes in. Databricks simplifies all these steps so that you can focus on building your final model, with the best prompts and performance.
# MAGIC
# MAGIC
# MAGIC ## GenAI & Maturity curve
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-maturity.png?raw=true" width="600px" style="float:right"/>
# MAGIC
# MAGIC Deploying GenAI can be done in multiple ways:
# MAGIC
# MAGIC - **Prompt engineering on public APIs (e.g. LLama 2, openAI)**: answer from public information, retail (think ChatGPT)
# MAGIC - **Retrieval Augmented Generation (RAG)**: specialize your model with additional content. *This is what we'll focus on in this demo*
# MAGIC - **OSS model Fine tuning**: when you have a large corpus of custom data and need specific model behavior (execute a task)
# MAGIC - **Train your own LLM**: for full control on the underlying data sources of the model (biomedical, Code, Finance...)
# MAGIC
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=%2F01-quickstart%2F00-RAG-chatbot-Introduction&demo_name=llm-rag-chatbot&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fllm-rag-chatbot%2F01-quickstart%2F00-RAG-chatbot-Introduction&version=1">
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## What is Retrieval Augmented Generation (RAG) for LLMs?
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-marchitecture.png?raw=true" width="700px" style="float: right" />
# MAGIC
# MAGIC RAG is a powerful and efficient GenAI technique that allows you to improve model performance by leveraging your own data (e.g., documentation specific to your business), without the need to fine-tune the model.
# MAGIC
# MAGIC This is done by providing your custom information as context to the LLM. This reduces hallucination and allows the LLM to produce results that provide company-specific data, without making any changes to the original LLM.
# MAGIC
# MAGIC RAG has shown success in chatbots and Q&A systems that need to maintain up-to-date information or access domain-specific knowledge.
# MAGIC
# MAGIC ### RAG and Vector Search
# MAGIC
# MAGIC To be able to provide additional context to our LLM, we need to search for documents/articles where the answer to our user question might be.
# MAGIC To do so,  a common solution is to deploy a vector database. This involves the creation of document embeddings, vectors of fixed size representing your document.<br/>
# MAGIC The vectors will then be used to perform real-time similarity search during inference.
# MAGIC
# MAGIC ### Implementing RAG with Databricks AI Foundation models
# MAGIC
# MAGIC In this demo, we will show you how to build and deploy your custom chatbot, answering questions on any custom or private information.
# MAGIC
# MAGIC As an example, we will specialize this chatbot to answer questions over Databricks, feeding databricks.com documentation articles to the model for accurate answers.
# MAGIC
# MAGIC Here is the flow we will implement:
# MAGIC
# MAGIC <!-- 
# MAGIC <div style="width: 400px; float: left; margin: 10px 20px 10px 10px; box-shadow: 0px 0px 10px #b5b5b5; padding:10px; min-height: 240px">
# MAGIC <h4 style="margin-left: 10px">1: Data prepration:</h4>
# MAGIC <ul>
# MAGIC   <li> Download databricks.com documentation articles</li>
# MAGIC   <li> Prepare the articles for our model (split into chunks)</li>
# MAGIC   <li> Compute the chunks embeddings using Databricks Foundation model (bge) and save them to a Delta table</li>
# MAGIC   <li> Add a Vector Search Index on our Delta table</li>
# MAGIC   </ul>
# MAGIC </div>
# MAGIC
# MAGIC <div style="width: 400px; float: left; margin: 10px; box-shadow: 0px 0px 10px #b5b5b5; padding:10px; min-height: 240px">
# MAGIC <h4 style="margin-left: 10px">2: Inferences:</h4>
# MAGIC <ul>
# MAGIC   <li>Build a langchain model using Databricks llama2-70 foundation model</li>
# MAGIC   <li>Retrieve simliar document from our Vector search index</li>
# MAGIC   <li>Deploy the chain using a Model Serving Endpoint</li>
# MAGIC </ul>
# MAGIC </div>
# MAGIC <br style="clear: both"> -->
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-self-managed-flow-0.png?raw=true" style="margin-left: 10px"  width="1100px;">

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC What you will learn:
# MAGIC
# MAGIC - How to extract information from unstructured documents (pdfs) and create custom chunks
# MAGIC - Leverage Databricks Embedding Foundation Model to compute the chunks embeddings
# MAGIC - Create a Self Managed Vector Search index and send queries to find similar documents
# MAGIC - How to use chains to support complex flow including chat history [using llama 2 input style]
# MAGIC - Deploy your Model Serving Endpoint with Table Inferences.
# MAGIC
# MAGIC Additional notebooks provided for:
# MAGIC - Evaluate your model chatbot model correctness with MLflow
# MAGIC - Run online llm evaluation and track your metrics with Databricks Monitoring

# COMMAND ----------

# MAGIC %run ../_resources/00-init-advanced 

# COMMAND ----------

dbutils.widgets.text("reset_all_data", "false", "Reset Data")
reset_all_data = dbutils.widgets.get("reset_all_data") == "true"

dbutils.widgets.text("dbName", 'rag_chat_lab')
dbName= dbutils.widgets.get("dbName")

dbutils.widgets.text("VECTOR_SEARCH_ENDPOINT_NAME", "workshop-vs-1")

VECTOR_SEARCH_ENDPOINT_NAME =  dbutils.widgets.get("VECTOR_SEARCH_ENDPOINT_NAME")

dbutils.widgets.text("dataset","Finance-SecFiles")


dataset = dbutils.widgets.get("dataset")

# COMMAND ----------

dbutils.widgets.text("catalog", 'genai_lab')
catalog= dbutils.widgets.get("catalog")


spark.sql(f"create catalog if not exists {catalog}")

spark.sql("use catalog " +catalog)

spark.sql(f"create DATABASE if NOT EXISTS `{dbName}`")

spark.sql("use " +dbName)

# COMMAND ----------

def reset_data(dbName):
  print(f'clearing up db {dbName}')
  spark.sql(f"DROP DATABASE IF EXISTS `{dbName}` CASCADE")
  spark.sql(f"create DATABASE if NOT EXISTS `{dbName}`")
  spark.sql(f"Use schema `{dbName}`")
  print(f'setting up the db {dbName}')

# COMMAND ----------

if reset_all_data:
  print("Resetting the Data and Models.")
  reset_data(dbName)
else:
  print("Not Resetting the Data and Models.")
