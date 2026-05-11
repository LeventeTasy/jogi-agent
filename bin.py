import os
import re
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # The Slay swap! 💅
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# 1. BEÁLLÍTÁSOK (Global Variables)
CHROMA_PATH = os.path.abspath(os.path.join(os.getcwd(), "chroma_db"))
DATA_PATH = os.path.abspath(os.path.join(os.getcwd(), "pdf"))


# 2. BEOLVASÁS
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


# 3. DARABOLÁS (A visszahozott Regex Mágia! 🪄✂️)
def split_documents(documents: list[Document]):
    all_chunks = []
    # A te zseniális regexed, ami megtalálja a cikkeket és paragrafusokat!
    para_pattern = r'(\d+\.\s§|\d+\.\s?[Cc]ikk|\(\d+\))'

    # 1. Csoportosítsuk az oldalakat forrásfájlonként 📂
    docs_by_source = {}
    for doc in documents:
        source = doc.metadata.get("source", "ismeretlen")
        if source not in docs_by_source:
            docs_by_source[source] = ""
        # Összefűzzük az egész dokumentumot egy nagy stringgé
        docs_by_source[source] += doc.page_content + "\n"

    # 2. Kisebb chunkok a biztonság kedvéért (Bye-bye hallucináció! 👋)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=40,
        length_function=len,
        is_separator_regex=False
    )

    for source, full_text in docs_by_source.items():
        # Szétszedjük a szöveget a regex minta alapján
        parts = re.split(para_pattern, full_text)
        current_article = "Általános"

        for i in range(len(parts)):
            part = parts[i].strip()
            if not part:
                continue

            # Ha a rész egyezik a mintával, akkor ez egy címsor! (pl. "7. cikk")
            if re.match(para_pattern, part):
                current_article = part
            else:
                # Ha nem, akkor ez a nyers szöveg, ezt vágjuk tovább LangChain-nel
                sub_chunks = text_splitter.split_text(part)
                for chunk_text in sub_chunks:
                    # BUMM! Itt tesszük bele az "article" metaadatot! 💥💎
                    new_doc = Document(
                        page_content=f"[{current_article}] {chunk_text}",
                        metadata={
                            "source": source,
                            "article": current_article,  # Itt a metaadat a JSON-höz!
                            "page": "N/A"  # Mivel összefűztük, az oldalszám már nem releváns
                        }
                    )
                    all_chunks.append(new_doc)

    return all_chunks


# 4. AZ "AGY" (Embeddings)
def get_embedding_function():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", api_key=api_key)
    return embeddings


# 5. ADATBÁZIS FELÉPÍTÉSE
def add_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Jelenlegi dokumentumok száma a DB-ben: {len(existing_ids)} 📚")

    last_article_id = None
    current_chunk_index = 0

    # ID-k kiosztása: most már az "article" alapján azonosítunk, nem az oldal alapján! 💅
    for chunk in chunks:
        source = chunk.metadata.get("source", "ismeretlen")
        article = chunk.metadata.get("article", "ismeretlen")
        current_article_id = f"{source}:{article}"

        if current_article_id == last_article_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_article_id}:{current_chunk_index}"
        chunk.metadata["id"] = chunk_id
        last_article_id = current_article_id

    new_chunks = []
    for chunk in chunks:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks) > 0:
        print(f"Új dokumentumok hozzáadása: {len(new_chunks)} db ✨")
        # Batching, nehogy a Google Rate Limit megint beszóljon! 🍕
        batch_size = 100
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i: i + batch_size]
            batch_ids = new_chunk_ids[i: i + batch_size]
            db.add_documents(batch, ids=batch_ids)
    else:
        print("Nincs új dokumentum, everything is up to date! 💅")


def query_rag(query_text: str):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    PROMPT_TEMPLATE = """
    Te egy jogi asszisztens vagy, aki KIZÁRÓLAG a megadott kontextus alapján válaszol.

    KONTEXT:
    {context}

    SZABÁLYOK:
    - CSAK a kontextusban szereplő információkat használd.
    - TILOS információt kitalálni.

    KÉRDÉS:
    {question}
    """

    results = db.similarity_search_with_score(query_text, k=3)
    context_text = "\n\n--\n\n".join(doc.page_content for doc, _score in results)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0
    )
    response_text = llm.invoke(prompt)

    # Itt most már az 'article' is kinyerhető a metaadatokból a CrewAI JSON-jéhez! ✨
    sources = [{"id": doc.metadata.get("id"), "article": doc.metadata.get("article")} for doc, _score in results]

    return f"VÁLASZ:\n {response_text.content} \nFORRÁSOK:\n {sources}"


# 6. A FŐFOLYAMAT (Slay Pipeline)
def main():
    print("RAG rendszer indul... ✨")

    test_questions = [
        # --- SZJA Törvény ---
        # "Ki jogosult a 25 év alatti fiatalok kedvezményére és meddig vehető igénybe?",
        # "Milyen szabályok vonatkoznak a családi kedvezményre? Mekkora az összege egy eltartott esetén?",

        # --- Polgári Törvénykönyv (Ptk.) ---
        # "Mik a szerződés érvénytelenségének általános esetei a Ptk. szerint?",
        # "Mi a különbség a kártérítés és a kártalanítás között a magyar magánjogban?",
        # "Hogyan jön létre egy érvényes adásvételi szerződés az új Ptk. alapján?",

        # --- GDPR (Adatvédelem) ---
        "Melyek az érintettek jogai a GDPR rendelet alapján? Sorolj fel legalább ötöt!",
        "Mit jelent az 'elfeledtetéshez való jog' (törléshez való jog) és mikor korlátozható?",
        "Milyen feltételek mellett tekinthető az adatkezeléshez adott hozzájárulás érvényesnek?"

        # --- Cross-topic (Összetettebb) ---
    ]

    # Csak teszteléshez ki is írhatod őket
    for q in test_questions:
        print(f"\n🔍 TESZTELÉS: {q}")
        print(query_rag(q))


if __name__ == "__main__":
    main()