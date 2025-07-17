# RAG as a Service (RAGaaS) Microservices Architecture

This document explains the architecture and logic of the RAG as a Service (RAGaaS) platform, designed to provide efficient and scalable Retrieval Augmented Generation capabilities.

## 1. High-Level Overview (For Higher-Level Users)

RAGaaS acts as a smart assistant that can answer your questions by looking up information within your own documents. Imagine having a vast library of documents (such as PDFs or DOCX files) and wanting to ask questions about their content. RAGaaS makes this possible by:

- **Ingesting Documents**: You upload your documents to RAGaaS. The system processes them, understands their content, and stores them in a searchable format. RAGaaS supports different ingestion versions (V1 and V2), which  signify different processing pipelines.
- **Organizing Information**: Documents are organized into **Applications** and **Collections**. An Application can be thought of as a project, while a Collection represents a specific set of documents within that project.
- **Answering Questions**: When you ask a question, RAGaaS quickly finds the most relevant information from your documents and uses a powerful language model to generate a precise answer, along with the sources it used.
- **Benchmarking**: RAGaaS includes a built-in tool to evaluate its performance, ensuring consistent accuracy and relevance in its answers.

### Conceptual Flow:

```mermaid
graph TD
    User[Your Team/User] -- Asks Questions & Uploads Docs --> RagaasController[RAGaaS Central Hub]

    subgraph "RAGaaS Central Hub"
        RagaasController -- Manages Apps/Collections/Docs --> InternalAPI[Internal API Endpoints]
        InternalAPI -- Triggers --> BenchmarkingEndpoint[Benchmarking Endpoint]
    end

    subgraph "Learning from Documents (Ingestion)"
        RagaasController -- Sends Documents for Processing --> IngestionService[Document Ingestion Team]
        IngestionService -- Sends Docs for Text Extraction --> ParserService[Parser as a Service]
        ParserService -- Returns Extracted Text --> IngestionService
        IngestionService -- Processes & Organizes (using AI Models for Embeddings) --> DataStorage[Document Library &amp; Index (Database &amp; Cloud Storage)]
    end

    subgraph "Answering Questions (Inference)"
        InferenceService -- Retrieves Initial Relevant Info --> DataStorage
        InferenceService -- Refines Relevance --> CrossEncoderService[Cross Encoder as a Service]
        CrossEncoderService -- Returns Reranked Info --> InferenceService
        InferenceService -- Asks the LLM for an Answer --> LLM[Large Language Model - The AI Brain]
        LLM -- Provides Answer --> InferenceService
        InferenceService -- Sends Answer Back --> RagaasController
        RagaasController -- Sends Your Question --> InferenceService
    end

    RagaasController -- Delivers Answers --> User
    BenchmarkingEndpoint -- Reports on Performance --> User
```

## 2. Core Microservices

The RAGaaS platform is built as a set of interconnected microservices, each responsible for a specific part of the overall functionality.


### 2.1. Component Explanation

- **RAGaaS Controller**:
  This is the central API gateway and orchestrator of the RAGaaS platform. It exposes endpoints for managing applications, collections, documents, and handling inference requests. It coordinates interactions with other microservices to fulfill user requests. The config_controller.yml file lists the routers for various functionalities handled by the controller, including applications, collections, documents, inference, and benchmarking.
  
   - **Applications Management:** Handled by the applications.py router, the controller allows for registering new applications in RAGaaS.
   - **Collections Management:** Defined in collections.py, the controller enables operations such as getting, creating, and deleting collections associated         with a specific application.
   - **Documents Management:** The documents.py router handles document ingestion, allowing users to upload new documents to a collection within an application. Notably, it defines separate API endpoints for v1 and v2 document ingestion, which determine which message queue the ingestion job is sent to.
   - **Inference Handling:** The inference.py router manages answering user questions by proxying requests to the Inference Service.
   - **Benchmarking:** The benchmarking.py router provides an endpoint to trigger the RAG benchmark script and return results, likely in an Excel file.        
- **Ingestion Service**: This service is responsible for processing the documents uploaded to RAGaaS.
     - When a document is uploaded via the RAGaaS Controller ( through /v1/documents or /v2/documents endpoints defined in documents.py), the Controller    publishes an IngestionJob message to a specific **Message Queue**.
     - The config_controller.yml file specifies distinct queues for each ingestion version: ingestion_v1_input_job_queue and ingestion_v2_input_job_queue. This allows for different ingestion pipelines (e.g., Ingestion Pipeline V1, Ingestion Pipeline V2) to consume jobs from their respective queues.  
     - These ingestion pipelines (defined in ingestion_pipeline.py and using ingestion_pipeline_components.py) process the raw documents. This processing typically       involves:
       
        - **Downloading the document** from a temporary location (usually S3).
        - **Parsing** the document to extract text.
        - **Chunking** the text into smaller, manageable pieces.
        - **Generating embeddings** (numerical representations) for each chunk.
        - **Storing the raw and processed documents** (e.g., chunked text) in **S3 Storage**.
        - **Storing metadata and embeddings** (vector representations) in the **Database**.
       
- **Inference Service:** When a user asks a question, the Controller forwards the query to the Inference Service.
     - This service first **retrieves relevant context chunks** (and their associated embeddings) from the **Database** based on the user's query.
     - It might also retrieve additional raw document content from **S3 Storage** if needed for specific use cases (e.g., showing original snippets).
     - The retrieved context chunks and the user's query are then fed to a **Large Language Model (LLM)**.
     - The LLM generates an answer based on the provided context.
     - The Inference Service then returns this answer, potentially along with the source context chunks, back to the RAGaaS Controller, which then sends it to the user. The inference_pipeline.py file defines the abstract interface for how this retrieval and generation process should occur.
       
- **Message Queues (MQ)**: These (e.g., ActiveMQ as suggested by config_controller.yml) act as intermediaries for asynchronous communication between services. This decoupling ensures that services can operate independently and reliably, even under heavy load. Specifically, they facilitate the hand-off of IngestionJob messages from the Controller to the Ingestion Service, allowing for parallel processing and robust error handling.
- **Database & S3 Storage:** These components serve as the persistent storage layer for RAGaaS.
     - The **Database** (PostgreSQL, configured in config_controller.yml) stores structured data such as application metadata, collection details, document metadata, document chunk embeddings, and job statuses. It's optimized for quick retrieval of specific document chunks based on queries.
     - **S3 Storage**  is used for storing the actual raw document files and potentially processed versions of the documents ( parsed text, chunked data) that are too large or unstructured for the relational database.

- **Benchmarking Service:** This component, as indicated by benchmark.py and benchmarking.py, is responsible for testing the performance of the RAGaaS system. It likely involves setting up benchmark environments, ingesting test documents, running queries against ground truth data, and generating reports to evaluate accuracy and efficiency. The benchmark.py file defines a BenchmarkJob class that handles the setup of the environment, document ingestion, and running of inference to evaluate the system.               

