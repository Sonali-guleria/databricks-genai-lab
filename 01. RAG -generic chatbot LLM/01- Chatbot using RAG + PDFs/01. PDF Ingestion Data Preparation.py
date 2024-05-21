# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC # 1. Ingesting and preparing PDF for LLM and Self-Managed Vector Search Embeddings
# MAGIC
# MAGIC ## In this example, we will focus on ingesting PDF documents as a source for our retrieval process. 
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-pdf-self-managed-0.png?raw=true" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC
# MAGIC There are two datasets that we can use for the workshop:
# MAGIC
# MAGIC * Databricks ebook PDFs from [Databricks resources page](https://www.databricks.com/resources) to our knowledge database.
# MAGIC * SEC filings 
# MAGIC
# MAGIC Here are all the detailed steps:
# MAGIC
# MAGIC - Use autoloader to load the binary PDFs into our first table. 
# MAGIC - Use the `unstructured` library  to parse the text content of the PDFs.
# MAGIC - Use `llama_index` or `Langchain` to split the texts into chuncks.
# MAGIC - Compute embeddings for the chunks.
# MAGIC - Save our text chunks + embeddings in a Delta Lake table, ready for Vector Search indexing.
# MAGIC
# MAGIC
# MAGIC Lakehouse AI not only provides state of the art solutions to accelerate your AI and LLM projects, but also to accelerate data ingestion and preparation at scale, including unstructured data like PDFs.
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=advanced/01-PDF-Advanced-Data-Preparation&demo_name=chatbot-rag-llm&event=VIEW">

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
dbutils.widgets.dropdown("reset_all_data", "true", ["false", "true"], "Reset Data")
dbutils.widgets.dropdown("dataset", "Finance-SecFilings", ["Finance-SecFilings", "Databricks-Documentation"],"Dataset")

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

# DBTITLE 1,Running the Setup
# MAGIC %run ./00-Setup $reset_all_data=$reset_all_data $dbName=$dbName $catalog=$catalog $dataset=$dataset $VECTOR_SEARCH_ENDPOINT_NAME=$VECTOR_SEARCH_ENDPOINT_NAME 

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC select current_catalog(), current_database();

# COMMAND ----------

print(f"We are using below for this lab: \n catalog: {catalog}\n Schema/Database: {dbName}\n VECTOR SEARCH ENDPOINT NAME: {VECTOR_SEARCH_ENDPOINT_NAME}\n The Dataset is: {dataset} \n\n If something is incorrect, please update in the 00. Setup and re-run this command.\n")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Ingestion Flow
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-pdf-self-managed-1.png?raw=true" style="float: right" width="650">
# MAGIC
# MAGIC
# MAGIC The Data Preparation for PDFs usually includes the steps below, and we are going to follow a similar flow.
# MAGIC <br>
# MAGIC </br>
# MAGIC * Ingest the PDFs as a Delta Lake table with the provided path urls. We'll use [Databricks Autoloader](https://docs.databricks.com/en/ingestion/auto-loader/index.html) to incrementally ingest new files, autoloader easily ingests our unstructured PDF data in binary format.
# MAGIC * OCR the images to text [***Note**: Your cluster will need a few extra libraries that you would typically install with a cluster init script.*]
# MAGIC * Extract text for each PDF file
# MAGIC * Use libraries to clean the text. This would include removing headers, footers, or any information that you do not want to consider.
# MAGIC * UDF functions can be used to scale the ingestion process. Using those, you can execute the ingestion + transformation in parallel instead of doing everything serially.
# MAGIC

# COMMAND ----------

if "Fin" in dataset:
  # Set the volume name for finance dataset
  vol_name = "volume_sec_filings"
  # Define the symbols for finance dataset
  symbols = {"AAPL", "TSLA"}
  ## Additional Symbols below:
             #, "AMZN", "META", "NFLX", "GOOG", "MSFT", "ADP", "FANG", "HOOD", "INTU", "MRNA", "MRVL", "OKTA", "PYPL", "VRSK"}

  # Define the files pattern for finance dataset
  files_pattern = f"/{{{','.join(symbols)}}}*.pdf"
else:
  # Set the volume name for databricks documentation dataset
  vol_name = "volume_databricks_documentation"
  # Define the files pattern for databricks documentation dataset
  files_pattern = "*.pdf"

# Extract the table prefix from the volume name
tab_prefix = vol_name.split("volume_")[1]

# Create the volume if it doesn't exist
spark.sql(f"CREATE VOLUME IF NOT EXISTS {vol_name}")


# COMMAND ----------

# DBTITLE 1,Use Volumes to store non-tabular data and govern Access 
# List our raw PDF docs
volume_folder =  f"/Volumes/{catalog}/{dbName}/{vol_name}"
# Let's upload some pdf files to our volume as example
upload_pdfs_to_volume(volume_folder+"/pdfs",dataset)

# COMMAND ----------

# DBTITLE 1,List of pdf files
display(dbutils.fs.ls(volume_folder+"/pdfs"))

# COMMAND ----------

# DBTITLE 1,Ingesting PDF files as binary format using Databricks cloudFiles (Autoloader)


# Reading the PDF files in an incremental fashion
df = (
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "BINARYFILE")
    .option("pathGlobFilter", "*.pdf")
    .load(f"dbfs:{volume_folder}/pdfs/{files_pattern}")
)

# Write the data as a Delta table
(
    df.writeStream.trigger(availableNow=True)
    .option("checkpointLocation", f"dbfs:{volume_folder}/checkpoints/{tab_prefix}_raw")
    .table(tab_prefix+"_raw")
    .awaitTermination()
)

# COMMAND ----------

# DBTITLE 1,Quick glance at the data; look at the column content 
# Display the first 2 records from the {tab_prefix}_raw table
display(spark.sql(f"SELECT * FROM {tab_prefix}_raw limit 2"))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC
# MAGIC ## Chunking?
# MAGIC
# MAGIC **Why?** 
# MAGIC
# MAGIC LLM models typically have a maximum input context length, and you won't be able to compute embeddings for very long texts.
# MAGIC In addition, the longer your context length is, the longer it will take for the model to provide a response.
# MAGIC
# MAGIC
# MAGIC For determining the chunking strategy, start with:
# MAGIC
# MAGIC *  How relevant is the context to the prompt?
# MAGIC * How much context/chunks can I fit within the model’s token limit?
# MAGIC * User behavior? Are they going to ask long queries?
# MAGIC * Do I need to pass this output to the next LLM? 
# MAGIC
# MAGIC **How?**
# MAGIC * 1:1
# MAGIC * 1:N
# MAGIC * many more
# MAGIC     - Sentence Segmentation: Breaking long paragraphs or documents into individual sentences.
# MAGIC     - Paragraph Segmentation: Breaking longer texts into smaller paragraphs or sections.
# MAGIC     - Tokenization: Breaking text into individual tokens or words.
# MAGIC     - Topic Modeling: Identifying key topics or themes within a large text and segmenting based on these themes.
# MAGIC     - Attention Mechanisms: Focusing attention on relevant parts of the input text while processing.
# MAGIC
# MAGIC
# MAGIC
# MAGIC *Document preparation is key for your model to perform well, and multiple strategies exist depending on your dataset:*
# MAGIC
# MAGIC - Split document into small chunks (paragraph, h2...)
# MAGIC - Truncate documents to a fixed length
# MAGIC - The chunk size depends on your content and how you'll be using it to craft your prompt. Adding multiple small doc chunks in your prompt might give different results than sending only a big one
# MAGIC - Split into big chunks and ask a model to summarize each chunk as a one-off job, for faster live inference
# MAGIC - Create multiple agents to evaluate each bigger document in parallel, and ask a final agent to craft your answer...
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Splitting our big documentation pages in smaller chunks
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/chunk-window-size.png?raw=true" style="float: right" width="700px">
# MAGIC <br/>
# MAGIC The PDFs can be very large, this is too long to be sent as the prompt to our model. 
# MAGIC
# MAGIC Even for this workshop, some PDFs are very large, with a lot of text. Some recent studies also suggest that bigger window size isn't always better, as the LLMs seem to focus on the beginning and end of your prompt.
# MAGIC
# MAGIC We'll extract the content and then use llama_index `SentenceSplitter`, and ensure that each chunk isn't bigger than 500 tokens. 
# MAGIC
# MAGIC
# MAGIC **The chunk size and chunk overlap depend on the use case and the PDF files.**
# MAGIC
# MAGIC *Remember that your prompt+answer should stay below your model max window size (4096 for llama2).*
# MAGIC
# MAGIC
# MAGIC
# MAGIC #### LLM Window size and Tokenizer
# MAGIC
# MAGIC The same sentence might return different tokens for different models. LLMs are shipped with a `Tokenizer` that you can use to count tokens for a given sentence (usually more than the number of words) (see [Hugging Face documentation](https://huggingface.co/docs/transformers/main/tokenizer_summary) or [OpenAI](https://github.com/openai/tiktoken))
# MAGIC
# MAGIC Make sure the tokenizer you'll be using here matches your model. We'll be using the `transformers` library to count llama2 tokens with its tokenizer. This will also keep our document token size below our embedding max size (1024).
# MAGIC
# MAGIC <br/>
# MAGIC <br style="clear: both">
# MAGIC <div style="background-color: #def2ff; padding: 15px;  border-radius: 30px; ">
# MAGIC   <strong>Information</strong><br/>
# MAGIC   Remember that the following steps are specific to your dataset. This is a critical part to building a successful RAG assistant.
# MAGIC   <br/> Always take time to manually review the chunks created and ensure that they make sense and contain relevant information.
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,To extract our PDF,  we'll need to setup libraries in our nodes
# For the production use case, install the libraries at your cluster level with an init script instead. 
install_ocr_on_nodes()

# COMMAND ----------

# MAGIC %md 
# MAGIC Let's start by extracting text from our PDF.

# COMMAND ----------

# DBTITLE 1,Transform pdf as text
from unstructured.partition.auto import partition
import re

def extract_doc_text(x : bytes) -> str:
  # Read files and extract the values with unstructured
  sections = partition(file=io.BytesIO(x))
  def clean_section(txt):
    txt = re.sub(r'\n', '', txt)
    return re.sub(r' ?\.', '.', txt)
  # Default split is by section of document, concatenate them all together because we want to split by sentence instead.
  return "\n".join([clean_section(s.text) for s in sections]) 

# COMMAND ----------

# DBTITLE 1,Trying our text extraction function with a single pdf file- Test Function
import io
import re
with requests.get('https://github.com/databricks-demos/dbdemos-dataset/blob/main/llm/databricks-pdf-documentation/Databricks-Customer-360-ebook-Final.pdf?raw=true') as pdf:
  doc = extract_doc_text(pdf.content)  
  print(doc)

# COMMAND ----------

# DBTITLE 1,Let's Scale!
# MAGIC %md
# MAGIC This looks great. We'll now wrap it with a text_splitter to avoid having too big pages, and create a Pandas UDF function to easily scale that across multiple nodes.
# MAGIC
# MAGIC *Note that our pdf text isn't clean. **To make it nicer, we could use a few extra LLM-based pre-processing steps, asking to remove unrelevant content like the list of chapters and to only keep the core text.***

# COMMAND ----------

from llama_index.langchain_helpers.text_splitter import SentenceSplitter
from llama_index import Document, set_global_tokenizer
from transformers import AutoTokenizer

# Reduce the arrow batch size as our PDF can be big in memory
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 10)

@pandas_udf("array<string>")
def read_as_chunk(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    #set llama2 as tokenizer to match our model size (will stay below BGE 1024 limit)
    set_global_tokenizer(
      AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    )
    #Sentence splitter from llama_index to split on sentences
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
    def extract_and_split(b):
      txt = extract_doc_text(b)
      nodes = splitter.get_nodes_from_documents([Document(text=txt)])
      return [n.text for n in nodes]

    for x in batch_iter:
        yield x.apply(extract_and_split)

# COMMAND ----------

# DBTITLE 1,Binary Content --> Textual Chunks 
table_name = tab_prefix+"_chunked_pdfs"


#Note that we need to enable Change Data Feed on the table to create the index
spark.sql(f"CREATE TABLE IF NOT EXISTS {table_name}(\
  chunk_id BIGINT GENERATED BY DEFAULT AS IDENTITY,\
  path STRING,\
  content_chunk STRING\
) TBLPROPERTIES (delta.enableChangeDataFeed = true)" )


# COMMAND ----------



(spark.readStream.table(tab_prefix+'_raw')
      .withColumn("content", F.explode(read_as_chunk("content")))
      .selectExpr('path', 'content as content_chunk')
  .writeStream
    .trigger(availableNow=True)
    .option("checkpointLocation", f'dbfs:{volume_folder}/checkpoints/{tab_prefix}_chunked_pdfs')
    .table(table_name).awaitTermination())

# COMMAND ----------

display(spark.sql(f"select * from {table_name} limit 10"))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC
# MAGIC #Vector Search 

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## What's required for our Vector Search Index
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/databricks-vector-search-type.png?raw=true" style="float: right" width="800px">
# MAGIC
# MAGIC Databricks provide multiple types of vector search indexes:
# MAGIC
# MAGIC - **Managed embeddings**: you provide a text column and endpoint name and Databricks synchronizes the index and embeddings with your Delta table. You can also choose to writ back the embeddings to a Delta table. This is highly recommended for text-based embeddings as it is not only simple but also sync all the changes from the source to embeddings and indexes automatically.
# MAGIC - **Self Managed embeddings**: you compute the embeddings and save them as a field of your Delta Table, Databricks will then synchronize the index. This is recommended if you have pre-existing embeddings and you only need to create/sync indexes automatically. This is also recommended if you are using any non-text data such as images and embeddings are created already.
# MAGIC - **Direct index**: when you want to use and update the index without having a Delta Table. The user is responsible for updating this table using the REST API or the Python SDK. This type of index cannot be created using the UI. You must use the REST API or the SDK.
# MAGIC
# MAGIC In this workshop, we will show you how to setup a **Self-managed Embeddings** index. 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **To cover the different  flavors of Databricks Vector Search Index, we will use Managed embeddings (Via UI and SDK). There is also an example for the Self-Managed embeddings at the end.**

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC There are two components of Databricks Vector Search:<br>
# MAGIC <br>
# MAGIC
# MAGIC * Compute called **`Vector Search endpoint`**, this is used to handle all aspects of VS including index building, index serving, index storage.
# MAGIC For the purpose of the workshop, the endpoint has been created already for you. If you are looking to create your own, please use the below code:
# MAGIC
# MAGIC ```
# MAGIC from databricks.vector_search.client import VectorSearchClient
# MAGIC vsc = VectorSearchClient()
# MAGIC
# MAGIC if not endpoint_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME):
# MAGIC     vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")
# MAGIC
# MAGIC wait_for_vs_endpoint_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME)
# MAGIC print(f"Endpoint named {VECTOR_SEARCH_ENDPOINT_NAME} is ready.")
# MAGIC ```
# MAGIC <br>
# MAGIC
# MAGIC * **Vector Search Indexes**: These are the computed indexes.

# COMMAND ----------

# DBTITLE 1,Create the Managed vector search 
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

#The table we'd like to index
source_table_fullname = f"{catalog}.{dbName}.{table_name}"

# Where we want to store our index
vs_index_fullname = f"{catalog}.{dbName}.{tab_prefix}_managed_vs_index"

if not index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
  print(f"\nCreating index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}... This will take a few moments...")

  vsc.create_delta_sync_index_and_wait(
    endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
    source_table_name=source_table_fullname, 
    index_name=vs_index_fullname,
    pipeline_type='TRIGGERED',
    primary_key="chunk_id",
    embedding_source_column="content_chunk", #providing the text column directly 
    embedding_model_endpoint_name="databricks-bge-large-en",  ## your embeddings model
    sync_computed_embeddings=True ## this will create a new Delta table that has the embeddings just been calculated
    )
  
  index = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
  index_path = index.describe().get("status").get("message").split("status: ")[1]
  print(f"\nVS index created and check the status here: {index_path}")
else:
  #Trigger a sync to update our vs content with the new data saved in the table
  index = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
  if index.describe().get("status").get("detailed_state")== "ONLINE_NO_PENDING_UPDATE":
    index.sync()
    print(f"\nSyncing index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}... This might take a few moments...")
  else:
    print("\n Unable to sync as index is not ready to be synced yet...")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Searching for similar content
# MAGIC
# MAGIC That's all we have to do. Databricks will automatically capture and synchronize new entries in your Delta Lake Table.
# MAGIC
# MAGIC Note that depending on your dataset size and model size, index creation can take a few seconds to start and index your embeddings.
# MAGIC
# MAGIC Let's give it a try and search for similar content.
# MAGIC
# MAGIC *Note: `similarity_search` also supports a filters parameter. This is useful to add a security layer to your RAG system: you can filter out some sensitive content based on who is doing the call (for example filter on a specific department based on the user preference).*

# COMMAND ----------


if "Fin" in dataset:
  question = "Market segment for Apple?"
else:
  question = "Databricks cost and performance metric?"

index = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME,vs_index_fullname)

results = index.similarity_search(
  query_text = question,
  columns=["path", "content_chunk"],
  num_results=2)

results

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Next step: Deploy our chatbot model with RAG
# MAGIC
# MAGIC We've seen how Databricks Lakehouse AI makes it easy to ingest and prepare your documents, and deploy a Self Managed Vector Search index on top of it with just a few lines of code and configuration.
# MAGIC
# MAGIC This simplifies and accelerates your data projects so that you can focus on the next step: creating your realtime chatbot endpoint with well-crafted prompt augmentation.
# MAGIC
# MAGIC Open the [02a- SEC chatbot- Context-Chain-Agents-Deploy]($./02a- SEC chatbot- Context-Chain-Agents-Deploy) notebook to create and deploy a chatbot endpoint for Finance Dataset
# MAGIC
# MAGIC Open the [02B-Context-Chain-Agents-Deploy <Databricks Assistant>]($./02B-Context-Chain-Agents-Deploy <Databricks Assistant>) notebook to create and deploy a chatbot endpoint for Finance Dataset
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Additional Content Below as take-away
# MAGIC
# MAGIC As explained above, Databricks has multiple flavors of Vector Search. We used the Managed indexes today where both the indexes and embeddings are created and synchronized by Databricks. Instead, if you have pre-existing embeddings that you want to use and only want to create indexes then you can choose *self-managed* delta sync version instead. Below is an example, using SDK. You can also create the same using the UI. 
# MAGIC
# MAGIC For this example, we will use [`BGE` embeddings Foundation Model](/ml/endpoints/databricks-bge-large-en)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Introducing Databricks BGE Embeddings Foundation Model endpoints
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-pdf-self-managed-4.png?raw=true" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC Foundation Models are provided by Databricks, and can be used out-of-the-box.
# MAGIC
# MAGIC Databricks supports several endpoint types to compute embeddings or evaluate a model:
# MAGIC - A **foundation model endpoint**, provided by databricks (ex: llama2-70B, MPT...)
# MAGIC - An **external endpoint**, acting as a gateway to an external model (ex: Azure OpenAI)
# MAGIC - A **custom**, fined-tuned model hosted on Databricks model service
# MAGIC
# MAGIC Open the [Model Serving Endpoint page](/ml/endpoints) to explore and try the foundation models.
# MAGIC
# MAGIC For this workshop, we will use the foundation model [`BGE` embeddings Foundation Model](/ml/endpoints/databricks-bge-large-en) and `llama2-70B` (chat). <br/><br/>
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/databricks-foundation-models.png?raw=true" width="600px" >

# COMMAND ----------

# DBTITLE 1,Important!
# MAGIC %md
# MAGIC ### This is self-managed example for Vector Search Using SDK
# MAGIC
# MAGIC Please note that the steps below for using embedding endpoint is only required if you want to compute and manage your own embeddings and you will use *self-managed embeddings* while creating vector indexes on Databricks. 
# MAGIC
# MAGIC To do so, we will have to first compute the embeddings of our chunks and save them as a Delta Lake table field as `array&ltfloat&gt`
# MAGIC

# COMMAND ----------

# DBTITLE 1,Using Databricks Foundation model BGE as embedding endpoint
from mlflow.deployments import get_deploy_client

# bge-large-en Foundation models are available using the /serving-endpoints/databricks-bge-large-en/invocations api. 
deploy_client = get_deploy_client("databricks")

## NOTE: if you change your embedding model here, make sure you change it in the query step too
embeddings = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": ["What is Apache Spark?"]})
pprint(embeddings)

# COMMAND ----------

# DBTITLE 1,Create the final databricks_pdf_documentation table containing chunks
table_name = tab_prefix+"_w_embeddings"

#Note that we need to enable Change Data Feed on the table to create the index
spark.sql(f"CREATE TABLE IF NOT EXISTS {table_name}(\
  id BIGINT GENERATED BY DEFAULT AS IDENTITY,\
  path STRING,\
  content_chunks STRING,\
  embedding ARRAY <FLOAT>\
) TBLPROPERTIES (delta.enableChangeDataFeed = true)" )

# COMMAND ----------

table_name

# COMMAND ----------

# MAGIC %md
# MAGIC ### Computing the chunk embeddings and saving them to our Delta Table
# MAGIC
# MAGIC The last step is to now compute an embedding for all our documentation chunks. Let's create an udf to compute the embeddings using the foundation model endpoint.
# MAGIC
# MAGIC *Note that this part would typically be setup as a production-grade job, running as soon as a new documentation page is updated. <br/> This could be setup as a Delta Live Table pipeline to incrementally consume updates.*

# COMMAND ----------

@pandas_udf("array<float>")
def get_embedding(contents: pd.Series) -> pd.Series:
    import mlflow.deployments
    deploy_client = mlflow.deployments.get_deploy_client("databricks")
    def get_embeddings(batch):
        #Note: this will fail if an exception is thrown during embedding creation (add try/except if needed) 
        response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": batch})
        return [e['embedding'] for e in response.data]

    # Splitting the contents into batches of 150 items each, since the embedding model takes at most 150 inputs per request.
    max_batch_size = 150
    batches = [contents.iloc[i:i + max_batch_size] for i in range(0, len(contents), max_batch_size)]

    # Process each batch and collect the results
    all_embeddings = []
    for batch in batches:
        all_embeddings += get_embeddings(batch.tolist())

    return pd.Series(all_embeddings)

# COMMAND ----------

(spark.readStream.table(tab_prefix+'_raw')
      .withColumn("content", F.explode(read_as_chunk("content")))
      .withColumn("embedding", get_embedding("content"))
      .selectExpr('path', 'content as content_chunks', 'embedding')
  .writeStream
    .trigger(availableNow=True)
    .option("checkpointLocation", f'dbfs:{volume_folder}/checkpoints/pdf_chunk_embeddings')
    .table(table_name).awaitTermination())

# COMMAND ----------

display(spark.sql(f"SELECT * from {table_name} limit 10"))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### Our dataset is now ready! Let's create our Self-Managed Vector Search Index.
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-pdf-self-managed-3.png?raw=true" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC Our dataset is now ready. We chunked the documentation pages into small sections, computed the embeddings and saved it as a Delta Lake table.
# MAGIC
# MAGIC Next, we'll configure Databricks Vector Search to ingest data from this table.
# MAGIC
# MAGIC Vector search index uses a Vector search endpoint to serve the embeddings (you can think about it as your Vector Search API endpoint). <br/>
# MAGIC Multiple Indexes can use the same endpoint. 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC For this workshop, we have already created Vector Search endpoints for you to use. **Please ensure** you are **choosing** one of the *vector index endpoint from the **list** provided by the **instructor**.*
# MAGIC
# MAGIC
# MAGIC There are several ways to create a vector Search endpoint. 
# MAGIC
# MAGIC * Using the UI [AWS](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-a-vector-search-endpoint-using-the-ui) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/create-query-vector-search#create-a-vector-search-endpoint)   
# MAGIC * Python SDK [AWS](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-a-vector-search-endpoint-using-the-python-sdk) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/create-query-vector-search#create-a-vector-search-endpoint-using-the-python-sdk)
# MAGIC * REST API [AWS](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-a-vector-search-endpoint-using-the-rest-api)| [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/create-query-vector-search#create-a-vector-search-endpoint-using-the-rest-api)
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC You can view your endpoint on the [Vector Search Endpoints UI](#/setting/clusters/vector-search). Click on the endpoint name to see all indexes that are served by the endpoint.

# COMMAND ----------

# DBTITLE 1,Create the Self-managed vector search using our endpoint
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()


#The table we'd like to index
source_table_fullname = f"{catalog}.{dbName}.{table_name}"
# Where we want to store our index
vs_index_fullname = f"{catalog}.{dbName}.{tab_prefix}_self_managed_vs_index"

if not index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
  print(f"\nCreating index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}... This will take a few moments...")
  vsc.create_delta_sync_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
    index_name=vs_index_fullname,
    source_table_name=source_table_fullname,
    pipeline_type="TRIGGERED", #Sync needs to be manually triggered
    primary_key="id",
    embedding_dimension=1024, #Match your model embedding size (bge)
    embedding_vector_column="embedding"
  )
else:
  #Trigger a sync to update our vs content with the new data saved in the table
  vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).sync()

#Let's wait for the index to be ready and all our embeddings to be created and indexed
wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### Catalog exploration for Indexes
# MAGIC
# MAGIC Navigate to your catalog and see if the indexes are ready! 

# COMMAND ----------

question = "What is the data about?"

response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": [question]})
embeddings = [e['embedding'] for e in response.data]

results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).similarity_search(
  query_vector=embeddings[0],
  columns=["path", "content_chunks"],
  num_results=1)
docs = results.get('result', {}).get('data_array', [])
pprint(docs)
