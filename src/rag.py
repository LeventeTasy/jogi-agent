import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # The Slay swap! 💅
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# 1. BEÁLLÍTÁSOK (Global Variables)
DATA_PATH = "../pdf"  # Ide tedd a Munka Törvénykönyvét!
CHROMA_PATH = "../chroma_db"  # Ennek a mappának a nevét muszáj megadni, ide menti az adatbázist!


# 2. BEOLVASÁS
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


# 3. DARABOLÁS (Chunking)
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)


# 4. AZ "AGY" (Embeddings)
def get_embedding_function():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    # Itt mondjuk meg neki, hogy a Google "szemüvegén" keresztül nézze a szöveget 👓
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", api_key=api_key)
    return embeddings


# 5. ADATBÁZIS FELÉPÍTÉSE (A rendrakás fázis)
def add_chroma(chunks: list[Document]):
    # Csatlakozunk az adatbázishoz
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Megnézzük, mi van már benne
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Jelenlegi dokumentumok száma a DB-ben: {len(existing_ids)} 📚")

    last_page_id = None
    current_chunk_index = 0

    # ID-k kiosztása minden egyes szeletnek (chunk)
    for chunk in chunks:
        source = chunk.metadata.get("source", "ismeretlen")
        page = chunk.metadata.get("page", "0")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Kiszámoljuk az egyedi ID-t és elmentjük a metadatába! ✨
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        chunk.metadata["id"] = chunk_id

        # Frissítjük a last_page_id-t a következő körhöz! (Ez nagyon hiányzott nálad 😭)
        last_page_id = current_page_id

    # Kiválogatjuk azokat, amik még nincsenek az adatbázisban
    new_chunks = []
    for chunk in chunks:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    # Hozzáadjuk a friss husit az adatbázishoz! 🥩
    if len(new_chunks) > 0:
        print(f"Új dokumentumok hozzáadása: {len(new_chunks)} db ✨")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        # db.persist() -> Régebbi verziókhoz kellett, de a LangChain Chromában ma már automatikus, ha persist_directory-t adsz meg!
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
        model="gemini-2.5-flash",  # A 2.5-flash lehet, hogy még túl új, a 1.5-flash a bombabiztos választás!
        temperature=0
    )
    response_text = llm.invoke(prompt)
    sources = [doc.metadata.get("id", None) for doc, _score in
               results]  # A results-ból szedjük a forrást, nem a válaszból!

    """
    print("\nVÁLASZ:")
    print(response_text.content)  # .content kell, mert a válasz egy objektum!

    print("\nFORRÁSOK:")
    print(sources)
    print("-" * 50)
    """

    return f"VÁLASZ:\n {response_text.content} \nFORRÁSOK:\n {sources}"

def rag_tool(query_text: str) -> str :
    documents = load_documents()
    if not documents:
        print("Girl, üres a mappa! Tegyél bele egy PDF-et! 💀")
        return
    chunks = split_documents(documents)
    add_chroma(chunks)

    return query_rag(query_text)

# 6. A FŐFOLYAMAT (Slay Pipeline)
def main():
    print("Indul a RAG építés... 🎉")

    # 1. Betöltjük a PDF-eket
    documents = load_documents()
    if not documents:
        print("Girl, üres a mappa! Tegyél bele egy PDF-et! 💀")
        return
    print(f"{len(documents)} oldal beolvasva. ✨")

    # 2. Felszeleteljük
    chunks = split_documents(documents)
    print(f"{len(chunks)} szeletre vágva. 🍕")

    # 3. Betoljuk az adatbázisba
    add_chroma(chunks)
    print("Adatbázis frissítve, te kész is vagy, queen! 👑")


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