from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader

openai.api_key = "your_api_key"

# Load document
loader = TextLoader("knowledge_base.txt")
documents = loader.load()
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(model_name="gpt-4"),
    retriever=vectorstore.as_retriever()
)

def answer_question(question):
    return qa_chain.run(question)

if __name__ == "__main__":
    question = input("Ask a question: ")
    print("Answer:", answer_question(question))
