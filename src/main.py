import os
import re
import time
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


# =========================
# 1. BEÁLLÍTÁSOK
# =========================
DATA_PATH = "../pdf"
CHROMA_PATH = "../chroma_db"
RETRIEVAL_K = 6


# =========================
# 2. SEGÉD FUNKCIÓK
# =========================
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def detect_law_name(source_path: str) -> str:
    filename = os.path.basename(source_path).replace(".pdf", "")

    if filename == "1995_CXVII_SZJA_TVK":
        return "SZJA törvény"
    if filename == "2013_V_PTK":
        return "Polgári Törvénykönyv (Ptk.)"
    if filename == "edutax_mt2026_web":
        return "Munka Törvénykönyve (Mt.)"
    if filename == "GDPR_2016":
        return "GDPR rendelet"

    return filename


def detect_section_type(section_id: str) -> str:
    section_id = section_id.strip()

    if re.fullmatch(r"\(\d+\)", section_id):
        return "preambulum"
    if "cikk" in section_id.lower():
        return "cikk"
    if "§" in section_id:
        return "paragrafus"

    return "ismeretlen"


def clean_id_text(text: str) -> str:
    return (
        text.replace(" ", "")
            .replace(".", "")
            .replace("(", "")
            .replace(")", "")
            .replace("§", "§")
    )


# =========================
# 3. DARABOLÁS
# =========================
def split_documents(documents: list[Document]):
    """
    Egy chunk = egy jogi egység.
    GDPR: cikk / preambulum
    Ptk., Mt., SZJA: § vagy cikk
    """
    all_chunks = []

    docs_by_source = {}
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        docs_by_source.setdefault(source, []).append(doc)

    # Egyetlen regex, de a találatokat jogi egységként kezeljük
    section_pattern = re.compile(
        r'(?m)^\s*((?:\d+:\d+|\d+)\.\s*§|\d+\.\s*§|\d+\.\s*[Cc]ikk|\(\d+\))\s*'
    )

    for source, pages in docs_by_source.items():
        pages.sort(key=lambda x: x.metadata.get("page", 0))
        full_text = "\n".join([p.page_content for p in pages])

        matches = list(section_pattern.finditer(full_text))

        # Ha nincs struktúra a szövegben, akkor egyben mentjük
        if not matches:
            law_name = detect_law_name(source)
            all_chunks.append(
                Document(
                    page_content=full_text.strip(),
                    metadata={
                        "source": source,
                        "law": law_name,
                        "section_type": "ismeretlen",
                        "section_id": "ismeretlen",
                        "page": "multiple",
                    },
                )
            )
            continue

        for i, match in enumerate(matches):
            section_id = match.group(1).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
            body = full_text[start:end].strip()

            if not body:
                continue

            section_type = detect_section_type(section_id)
            law_name = detect_law_name(source)

            chunk = Document(
                page_content=f"{section_id}\n{body}",
                metadata={
                    "source": source,
                    "law": law_name,
                    "section_type": section_type,
                    "section_id": section_id,
                    "page": "multiple",
                },
            )
            all_chunks.append(chunk)

    return all_chunks


# =========================
# 4. EMBEDDING
# =========================
def get_embedding_function():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        api_key=api_key
    )


# =========================
# 5. CHROMA FELÉPÍTÉS
# =========================
def add_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Jelenlegi dokumentumok száma a DB-ben: {len(existing_ids)} 📚")

    new_chunks = []

    for idx, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "ismeretlen")
        law = chunk.metadata.get("law", "ismeretlen_törvény")
        section_type = chunk.metadata.get("section_type", "ismeretlen")
        section_id = chunk.metadata.get("section_id", "ismeretlen")

        clean_source = os.path.basename(source).replace(".pdf", "")
        clean_law = clean_id_text(law)
        clean_section_id = clean_id_text(section_id)

        chunk_id = f"{clean_source}:{clean_law}:{section_type}:{clean_section_id}:{idx}"
        chunk.metadata["id"] = chunk_id

        if chunk_id not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks) == 0:
        print("Minden friss, nincs mit tölteni! 💅")
        return

    print(f"Új dokumentumok hozzáadása: {len(new_chunks)} db ✨")

    batch_size = 100
    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i:i + batch_size]
        batch_ids = [chunk.metadata["id"] for chunk in batch]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                db.add_documents(batch, ids=batch_ids)
                print(f"Adag elküldve: {min(i + batch_size, len(new_chunks))}/{len(new_chunks)}... ✅")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 10 * (attempt + 1)
                    print(f"Hiba történt (attempt {attempt + 1})... újrapróbálom {wait_time} mp múlva...")
                    time.sleep(wait_time)
                else:
                    print(f"Végzetes hiba: {e}")
                    raise

        time.sleep(2)


# =========================
# 6. KÉRDEZÉS
# =========================
def query_rag(query_text: str):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    PROMPT_TEMPLATE = """
    Te egy tűpontos jogi asszisztens vagy. KIZÁRÓLAG a megadott kontextus alapján válaszolj.
    
    SZABÁLYOK:
    - Csak a kontextusban szereplő információkat használd.
    - Tilos bármit kitalálni vagy feltételezni.
    - Csak olyan törvényt, cikket, paragrafust vagy preambulum bekezdést említs, ami szó szerint szerepel a kontextusban.
    - A preambulum nem azonos a cikkel vagy paragrafussal.
    - Ha a pontos hivatkozás nem állapítható meg, ezt mondd ki egyértelműen.
    - Maximum 2-3 releváns forrást említs.
    - Ha a kérdés jogokat, kötelezettségeket vagy definíciókat kér, elsődlegesen cikket vagy paragrafust használj, ne preambulumot.
    
    Ha nincs explicit cikk a contextben:
    - NE írj cikket
    - NE próbáld behelyettesíteni
    - csak: "nem állapítható meg"
    
    HA egy cikk vagy paragrafus nem szerepel szó szerint a kontextusban:
    - TILOS megemlíteni
    - TILOS számot kitalálni
    
    KONTEXTUS:
    {context}
    
    KÉRDÉS:
    {question}
    
    VÁLASZ FORMÁTUMA:
    - Rövid, pontos válasz
    - Ha lehet: "A [Törvény neve] [cikk/paragrafus] alapján..."
    - Ha nem biztos a pontos hivatkozás:
      "A rendelkezésre álló szöveg alapján a pontos hivatkozás nem állapítható meg, de..."
    - A végén: "Források:"
    """

    results = db.similarity_search_with_score(
        query_text,
        k=RETRIEVAL_K,
        filter={"section_type": {"$in": ["cikk", "paragrafus"]}}
    )

    context_parts = []
    source_list = []

    for doc, score in results:
        source_name = os.path.basename(doc.metadata.get("source", "Ismeretlen")).replace(".pdf", "")
        law = doc.metadata.get("law", "Ismeretlen törvény")
        section_type = doc.metadata.get("section_type", "ismeretlen")
        section_id = doc.metadata.get("section_id", "ismeretlen")

        label = f"[{law} | {section_type} | {section_id} | {source_name}]"
        context_parts.append(f"{label}\n{doc.page_content}")

        source_list.append({
            "id": doc.metadata.get("id", None),
            "law": law,
            "section_type": section_type,
            "section_id": section_id,
            "score": score
        })

    context_text = "\n\n---\n\n".join(context_parts)

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    response_text = llm.invoke(prompt)

    print("\nVÁLASZ:")
    print(response_text.content)

    print("\nFORRÁSOK:")
    for src in source_list:
        print(f"- {src['law']} | {src['section_type']} | {src['section_id']} | score={src['score']:.4f}")

    print("-" * 50)


# =========================
# 7. FŐFOLYAMAT
# =========================
def main():
    print("Indul a RAG építés... 🎉")

    documents = load_documents()
    if not documents:
        print("Üres a mappa! Tegyél bele egy PDF-et!")
        return

    print(f"{len(documents)} oldal beolvasva. ✨")

    chunks = split_documents(documents)
    print(f"{len(chunks)} szeletre vágva.")

    add_chroma(chunks)
    print("Adatbázis frissítve, te kész is vagy, queen!")

    test_questions = [
        # --- GDPR jogok (csapda: túl általános + összekeverhető cikkek) ---
        "Felsorolható-e a GDPR alapján az 'információhoz való jog' mint önálló érintetti jog, és melyik cikk szabályozza pontosan?",
        "Az adathordozhatósághoz való jog minden adatkezelési jogalap esetén érvényesül?",
        "A GDPR szerint a hozzájárulás visszavonása érinti-e a korábbi adatkezelés jogszerűségét?",

        # --- elfeledtetés (csapda: túl széles / kivételek / jogalap keverés) ---
        "Az elfeledtetéshez való jog automatikusan alkalmazandó minden adatkezelés esetén?",
        "Ha egy adatot közérdekből kezelnek, akkor kérhető-e annak törlése a GDPR szerint?",
        "A törléshez való jog és az adatkezelés korlátozása ugyanazt jelenti-e a GDPR-ban?",

        # --- hozzájárulás (csapda: definíció vs feltételek keverése) ---
        "Elég-e a GDPR szerint az, ha a felhasználó nem tiltakozik az adatkezelés ellen, hogy az hozzájárulásnak minősüljön?",
        "A GDPR szerint mindig érvénytelen a hozzájárulás, ha szolgáltatás igénybevételéhez kötik?",
        "Egy előre kipipált checkbox elfogadható hozzájárulásnak minősülhet valaha a GDPR szerint?",

        # --- Mt + GDPR keverés (csapda: rossz jogalap / túl specifikus állítások) ---
        "A munkáltató a GDPR alapján bármilyen személyes adatot kérhet a munkavállalótól, ha az a munkavégzéshez kapcsolódik?",
        "A biometrikus adatok kezelése a Munka Törvénykönyve szerint mindig megengedett a beléptető rendszerekhez?",
        "A munkavállaló hozzájárulása elegendő jogalap-e minden munkaviszonnyal kapcsolatos adatkezeléshez?",
        "A GDPR 88. cikk teljes mértékben felülírja a magyar Munka Törvénykönyv adatkezelési szabályait?"
    ]

    for question in test_questions:
        print(f"\n🔍 TESZTELÉS: {question}")
        query_rag(question)

    query_text = input(": ")
    while query_text != "break":
        query_rag(query_text)
        query_text = input(": ")


if __name__ == "__main__":
    main()