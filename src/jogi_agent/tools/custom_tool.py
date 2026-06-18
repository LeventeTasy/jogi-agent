import os
from typing import Type
from pydantic import BaseModel, Field, PrivateAttr
from crewai.tools import BaseTool
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from rag import build_rag


class JogiKeresoInput(BaseModel):
    """Input schema for MyCustomTool."""
    # Átneveztem query-re, mert az ágensek jobban értik, mit kell ide írni! 🎯
    query: str = Field(..., description="A jogi kérdés vagy keresőszavak.")


class MyCustomTool(BaseTool):
    name: str = "Jogi adatbázis"
    description: str = (
        "Használd ezt a szerszámot, ha a Munka Törvénykönyve, Ptk vagy GDPR kapcsán kell keresned."
    )
    args_schema: Type[BaseModel] = JogiKeresoInput

    _db: Chroma = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        build_rag()

        # Elérési út, embeddings beállítása
        db_path = os.path.abspath(os.path.join(os.getcwd(), "../chroma_db"))
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

        # Mentés a belső _db változóba
        self._db = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
        print("--- RAG Adatbázis sikeresen betöltve a memóriába! ---")

    def _run(self, query: str) -> str:
        # A self._db-ből kérünk ki top 10 találatot)
        results = self._db.similarity_search_with_score(
            query,
            k=10,
            filter={"section_type": {"$in": ["cikk", "paragrafus"]}}
        )

        formatted_context = ""


        for doc, _score in results:
            meta = doc.metadata
            law = meta.get("law", "Ismeretlen törvény")
            sid = meta.get("section_id", "N/A")
            page = meta.get("page", "multiple")
            src = os.path.basename(meta.get("source", "file"))

            distance_score = round(float(_score), 4)

            clean_content = doc.page_content[:1500] + "... [vágva a stabilitásért]" if len(
                doc.page_content) > 1500 else doc.page_content

            header = f"[Törvény: {law} | Hely: {sid} | Távolság: {distance_score} | Oldal: {page} | Forrásfájl: {src}]"
            formatted_context += f"{header}\n{clean_content}\n\n---\n\n"

        return formatted_context