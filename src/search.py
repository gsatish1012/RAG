import os
import re
from dotenv import load_dotenv
from vectorstore import FaissVectorStore
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "llama-3.1-8b-instant"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        groq_api_key = os.getenv("GORK_API_KEY")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        # Split query into sub-queries for compound queries
        sub_queries = [q.strip() for q in re.split(r'\s*(?:and|,)\s*', query) if q.strip()]
        
        local_contexts = []
        web_queries = []
        
        for sub_query in sub_queries:
            results = self.vectorstore.query(sub_query, top_k=top_k)
            distances = [r["distance"] for r in results]
            print(f"Distances for '{sub_query}': {distances}")
            if results and min(distances) <= 1.2:  # Relevant locally
                texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
                local_contexts.append("\n\n".join(texts))
            else:  # Not relevant, add to web search
                web_queries.append(sub_query)
        
        context = "\n\n".join(local_contexts)
        
        if web_queries:
            print(f"[INFO] Searching web for: {' and '.join(web_queries)}")
            web_results = DuckDuckGoSearchRun().run(' and '.join(web_queries))
            context += "\n\n" + web_results
        
        if not context:
            return "No relevant information found."
        
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke([prompt])
        return response.content

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "Explain about : HSTRN , RAG?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)