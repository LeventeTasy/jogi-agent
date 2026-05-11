from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import tool
from rag import build_rag

class JogiKeresoInput(BaseModel):
    """Input schema for JogiKeresoInput."""
    argument: str = Field(..., description="A jogi kérdés.")

class MyCustomTool(BaseTool):
    name: str = "Jogi adatbézis"
    description: str = (
        "Használd ezt a szerszámot, ha a Munka Törvénykönyve, Ptk vagy GDPR kapcsán kell keresned."
    )
    args_schema: Type[BaseModel] = JogiKeresoInput

    _db: Chroma = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        build_rag()

        db_path = os.path.abspath(os.path.join(os.getcwd(), "..\chroma_db"))
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

        self._db = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
        print("--- RAG Adatbázis sikeresen betöltve a memóriába! ---")

    def _run(self, query: str) -> str:
        results = self._db.similarity_search(query, k=4)

        formatted_context = ""
        for doc in results:
            sources = [{"id": doc.metadata.get("id"), "article": doc.metadata.get("article")} for doc, _score in results]

            splitted = name.split(":")
            source = splitted[0]
            page = splitted[1][1:]
            article = splitted[2]

            formatted_context += f"[Source: {source}, Page: {page} Article: {sources["article"]}]: {doc.page_content}\n\n"

        return formatted_contex
