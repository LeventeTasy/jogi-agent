import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # The Slay swap! 💅
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import re

# 1. BEÁLLÍTÁSOK (Global Variables)
DATA_PATH = "../pdf"  # Ide tedd a Munka Törvénykönyvét!
CHROMA_PATH = "../chroma_db"  # Ennek a mappának a nevét muszáj megadni, ide menti az adatbázist!

CHROMA_PATH = os.path.abspath(os.path.join(os.getcwd(), "../chroma_db"))
DATA_PATH = os.path.abspath(os.path.join(os.getcwd(), "../pdf"))

# 2. BEOLVASÁS
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


# 3. DARABOLÁS (Chunking)
def split_documents(documents: list[Document]):
    all_chunks = []
    # A te zseniális regexed
    para_pattern = r'(\d+\.\s§|\d+\.\s?[Cc]ikk|\(\d+\))'

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=40,
        length_function=len,
        is_separator_regex=False
    )

    current_article = "Általános"  # Ez "emlékszik" a cikkre oldalakon át is ✨

    for doc in documents:
        full_text = doc.page_content
        source = doc.metadata.get("source", "ismeretlen")
        page = doc.metadata.get("page", "?")  # Itt rántjuk vissza az oldalszámot! 📄✨

        # Szétszedjük az adott oldal szövegét
        parts = re.split(para_pattern, full_text)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if re.match(para_pattern, part):
                current_article = part  # Frissítjük a cikk számát, ha találtunk újat
            else:
                sub_chunks = text_splitter.split_text(part)
                for chunk_text in sub_chunks:
                    # Itt jön a metaadat-glow up! 💖
                    new_doc = Document(
                        page_content=f"[{current_article}] {chunk_text}",
                        metadata={
                            "source": source,
                            "article": current_article,
                            "page": page + 1,  # A PDF-ben 0-tól indul, mi 1-től mutatjuk (user-friendly!)
                            "id": f"{os.path.basename(source)}:p{page + 1}:{current_article}"  # Ütős egyedi ID
                        }
                    )
                    all_chunks.append(new_doc)

    return all_chunks


# 4. AZ "AGY" (Embeddings)
def get_embedding_function():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    # Itt mondjuk meg neki, hogy a Google "szemüvegén" keresztül nézze a szöveget 👓
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", api_key=api_key)
    return embeddings


# 5. ADATBÁZIS FELÉPÍTÉSE (A rendrakás fázis)
def add_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # 1. Lekérjük a már bent lévő ID-kat
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Jelenlegi dokumentumok száma a DB-ben: {len(existing_ids)} 📚")

    # 2. Csak azokat a chunkokat tartjuk meg, amiknek az ID-ja még nincs bent
    new_chunks = []
    new_chunk_ids = []

    for chunk in chunks:
        # A split_documents-ben már beállítottuk a chunk.metadata["id"]-t! ✨
        chunk_id = chunk.metadata.get("id")

        if chunk_id not in existing_ids:
            new_chunks.append(chunk)
            new_chunk_ids.append(chunk_id)

    # 3. Ha van új "doksi", betoljuk batch-elve 🍕
    if len(new_chunks) > 0:
        print(f"Új dokumentumok hozzáadása: {len(new_chunks)} db ✨")

        batch_size = 100
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i: i + batch_size]
            batch_ids = new_chunk_ids[i: i + batch_size]
            db.add_documents(batch, ids=batch_ids)
            print(f"Batch elküldve: {i + len(batch)}/{len(new_chunks)} ✅")
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
    
    ---
    
    SZABÁLYOK (kritikus):
    - CSAK a fenti kontextusban szereplő információkat használhatod
    - TILOS bármilyen információt kitalálni vagy feltételezni
    - Csak olyan paragrafust / cikket hivatkozhatsz, ami konkrétan szerepel a kontextusban
    - Ha nem egyértelmű a pontos cikk vagy paragrafus, MONDD KI hogy nem állapítható meg
    - Különböztesd meg:
      - jogszabályi cikkek / paragrafusok
      - preambulum bekezdések (pl. (32))
    - NE keverd ezeket össze
    - Minden állítást támassz alá forrással
    - Ha több különböző forrás ellentmond egymásnak, jelezd az ellentmondást
    
    FORMÁTUM:
    - Adj egy rövid, pontos választ
    - Utána: "Források:" rész
    - Maximum 2-3 releváns hivatkozás
    
    ---
    
    KÉRDÉS:
    {question}
    """

    results = db.similarity_search_with_score(query_text, k=3)
    context_text = "\n\n--\n\n".join(doc.page_content for doc, _score in results)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )
    response_text = llm.invoke(prompt)

    # Itt most már az 'article' is kinyerhető a metaadatokból a CrewAI JSON-jéhez! ✨
    sources = [{"id": doc.metadata.get("id"), "article": doc.metadata.get("article")} for doc, _score in results]

    """
    print("\nVÁLASZ:")
    print(response_text.content)  # .content kell, mert a válasz egy objektum!

    print("\nFORRÁSOK:")
    print(sources)
    print("-" * 50)
    """

    return f"VÁLASZ:\n {response_text.content} \nFORRÁSOK:\n {sources}"

def build_rag():
    print("Rag építése elkezdődött!")
    documents = load_documents()
    if not documents:
        print("A mappa üres, töltse fel PDF-ekkel!")
        return

    print(f"{len(documents)} oldal beolvasva. ✨")

    try:
        chunks = split_documents(documents)
        print(f"{len(chunks)} szeletre vágva. 🍕")
        add_chroma(chunks)
        print("Adatbázis frissítve, te kész is vagy, queen! 👑")
        print("A RAG build sikeresen befejeződött!")
    except Exception as e:
        print(f"Hiba {e}")

# 6. A FŐFOLYAMAT (Slay Pipeline)
def main():
    print(CHROMA_PATH)
    print(DATA_PATH)
    build_rag()


    print("RAG rendszer indul... ✨")

    #query_text = input("Kérdés (vagy 'break' a kilépéshez): ")

    teszt_kerdesek = ["Hány nap a felmondási időm, ha 3 éve dolgozom a cégnél és a munkáltató mond fel nekem?",
                      "Kiadhatja-e a főnököm a szabadságomat a próbaidő alatt, vagy meg kell várnom a 3 hónapot?",
                      "Elmehetek-e egy konkurens céghez dolgozni azonnal, ha aláírtam egy versenytilalmi megállapodást? Mennyit kell fizetniük érte?"]

    test_questions = [
        # --- SZJA Törvény ---
        "Ki jogosult a 25 év alatti fiatalok kedvezményére és meddig vehető igénybe?",
        "Milyen szabályok vonatkoznak a családi kedvezményre? Mekkora az összege egy eltartott esetén?",

        # --- Polgári Törvénykönyv (Ptk.) ---
        "Mik a szerződés érvénytelenségének általános esetei a Ptk. szerint?",
        "Mi a különbség a kártérítés és a kártalanítás között a magyar magánjogban?",
        "Hogyan jön létre egy érvényes adásvételi szerződés az új Ptk. alapján?",

        # --- GDPR (Adatvédelem) ---
        "Melyek az érintettek jogai a GDPR rendelet alapján? Sorolj fel legalább ötöt!",
        "Mit jelent az 'elfeledtetéshez való jog' (törléshez való jog) és mikor korlátozható?",
        "Milyen feltételek mellett tekinthető az adatkezeléshez adott hozzájárulás érvényesnek?",

        # --- Cross-topic (Összetettebb) ---
        "Hogyan kell kezelni a munkavállaló adatait a munkaviszony során a GDPR és az Mt. szerint?"
    ]

    """
    # Ezt csak dobd bele egy for ciklusba és mehet a query_rag()! 🚀
    for question in test_questions:
        print(f"\n🔍 TESZTELÉS: {question}")
        query_rag(question)
    """

    query_text = input(": ")

    while query_text != "break":
        print(query_rag(query_text))
        query_text = input(": ")

if __name__ == "__main__":
    main()