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
        results = db.similarity_search_with_score(
            query_text,
            k=6,
            filter={"section_type": {"$in": ["cikk", "paragrafus"]}}
        )

        formatted_context = ""
        for doc in results:

            sources = [
                {
                    "source": os.path.basename(doc.metadata.get("source", "ismeretlen")),
                    "law": doc.metadata.get("law", "N/A"),
                    "section_type": doc.metadata.get("section_type", "N/A"),
                    "section_id": doc.metadata.get("section_id", "N/A"),
                    "page": doc.metadata.get("page", "N/A")
                }
                for doc, _score in results
            ]

            formatted_context = ""
            for i, (doc, _score) in enumerate(results):
                # Kinyerjük a metaadatokat az adott dokumentumból
                meta = doc.metadata
                law = meta.get("law", "Ismeretlen törvény")
                sid = meta.get("section_id", "N/A")
                page = meta.get("page", "multiple")
                src = os.path.basename(meta.get("source", "file"))
                conf = round(1 - min(score, 1.0), 2)

                header = f"[Törvény: {law} | Hely: {sid} | Confidence: {conf} | Oldal: {page} | Forrásfájl: {src}]"

                formatted_context += f"{header}\n{doc.page_content}\n\n---\n\n"

        return formatted_contex
