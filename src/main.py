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
import time

# 1. BEÁLLÍTÁSOK (Global Variables)
DATA_PATH = "../pdf"  # Ide tedd a Munka Törvénykönyvét!
CHROMA_PATH = "../chroma_db"  # Ennek a mappának a nevét muszáj megadni, ide menti az adatbázist!


# 2. BEOLVASÁS
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


# 3. DARABOLÁS (Chunking)
def split_documents(documents: list[Document]):
    all_chunks = []

    # Ez a minta keresi meg a paragrafusokat (pl. "64. §")
    # A zárójel azért kell, hogy a re.split ne dobja el a mintát, hanem tartsa meg!
    para_pattern = r'(\d+\.\s§)'

    for doc in documents:
        full_text = doc.page_content
        source = doc.metadata.get("source")
        page = doc.metadata.get("page")

        # Szétvágjuk a szöveget a paragrafus jelek mentén
        parts = re.split(para_pattern, full_text)

        # A re.split eredménye egy lista lesz: ['', '1. §', 'Szöveg...', '2. §', 'Szöveg...']
        # Az első elem gyakran üres, ha a szöveg rögtön paragrafussal kezdődik.

        current_para = "Ismeretlen §"

        # Bejárjuk a darabokat és összerakjuk a címet a tartalommal
        for i in range(len(parts)):
            part = parts[i].strip()
            if not part:
                continue

            # Ha a darab egy paragrafus jelzés (pl. "64. §")
            if re.match(para_pattern, part):
                current_para = part
            else:
                # Ez maga a szöveges tartalom a következő paragrafusig
                # Létrehozunk egy új Document objektumot
                new_chunk = Document(
                    page_content=f"{current_para} {part}",
                    metadata={
                        "source": source,
                        "page": page,
                        "paragrafus": current_para  # Itt a lényeg! 👑
                    }
                )
                all_chunks.append(new_chunk)

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
        para = chunk.metadata.get("paragrafus", "ismeretlen_para")

        current_page_id = f"{source}:{page}"

        # Számoljuk, hányadik darab ez az adott oldalon
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Tisztítjuk a paragrafus nevet 💅
        clean_para = para.replace(" ", "").replace(".", "")

        # THE MAGIC: Hozzácsapjuk az indexet is a végére! So no more matching outfits! 👯‍♀️🚫
        chunk_id = f"{current_page_id}:{clean_para}:{current_chunk_index}"
        chunk.metadata["id"] = chunk_id

        # Frissítjük a last_page_id-t a következő körhöz!
        last_page_id = current_page_id

    # Kiválogatjuk azokat, amik még nincsenek az adatbázisban
    new_chunks = []
    for chunk in chunks:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)


    # Hozzáadjuk a darabokat kisebb adagokban (batching)
    if len(new_chunks) > 0:
        print(f"Új dokumentumok hozzáadása: {len(new_chunks)} db ✨")

        batch_size = 100
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i + batch_size]
            batch_ids = [chunk.metadata["id"] for chunk in batch]

            # Itt jön a TRY-CATCH (Pythonul: try-except) trükk! 🕵️‍♀️🔥
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    db.add_documents(batch, ids=batch_ids)
                    print(f"Adag elküldve: {min(i + batch_size, len(new_chunks))}/{len(new_chunks)}... ✅")
                    break  # Ha sikerült, ugrunk a következő adagra!
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 10 * (attempt + 1)
                        print(
                            f"Hiba történt (attempt {attempt + 1})... Sebaj, Queen! 💅 Pihi {wait_time}mp és újrapróbálom...")
                        time.sleep(wait_time)
                    else:
                        print(f"Végzetes hiba! A Google ma nagyon nem akarja... 💀 Error: {e}")
                        raise e  # Ha 3-szor is elbukott, akkor engedjük el

            # Egy kis fix pihenő minden sikeres batch után, hogy ne kapjunk 429-est 🧘‍♀️
            time.sleep(2)
    else:
        print("Minden friss, nincs mit tölteni! 💅")


def query_rag(query_text: str):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    PROMPT_TEMPLATE = """
        Te egy elegáns és tűpontos munkajogi és adójogi asszisztens vagy. 
        A válaszodban MINDIG nevezd meg a törvényt és a pontos paragrafust!
        Használd a forrásfájl nevét a törvény azonosításához:
        - 1995_CXVII_SZJA_TVK -> SZJA törvény
        - 2013_V_PTK -> Polgári Törvénykönyv (Ptk.)
        - edutax_mt2026_web -> Munka Törvénykönyve (Mt.)
        - GDPR_2016 -> GDPR rendelet
    
        KONTEXTUS:
        {context}
    
        ---
        KÉRDÉS: {question}
    
        VÁLASZ (Formátum: 'A [Törvény neve] [X. §]-a alapján...'):
        """

    results = db.similarity_search_with_score(query_text, k=10)

    context_parts = []
    for doc, _score in results:
        # Kiszedjük a fájlnevet a metadata-ból és levágjuk az elérési utat meg a kiterjesztést (.pdf)
        source_name = os.path.basename(doc.metadata.get("source", "Ismeretlen")).replace(".pdf", "")
        # Összerakjuk: [Fájlnév]: [Szöveg]
        context_parts.append(f"FORRÁS ({source_name}):\n{doc.page_content}")

    context_text = "\n\n---\n\n".join(context_parts)

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    response_text = llm.invoke(prompt)
    print("\nVÁLASZ:")
    print(response_text.content)  # .content kell, mert a válasz egy objektum!

    print("\nFORRÁSOK:")
    sources = [doc.metadata.get("id", None) for doc, _score in
               results]  # A results-ból szedjük a forrást, nem a válaszból!
    print(sources)
    print("-" * 50)

# 6. A FŐFOLYAMAT (Slay Pipeline)
def main():
    print("Indul a RAG építés... 🎉")

    # 1. Betöltjük a PDF-eket
    documents = load_documents()
    if not documents:
        print("Üres a mappa! Tegyél bele egy PDF-et!")
        return
    print(f"{len(documents)} oldal beolvasva. ✨")

    # 2. Felszeleteljük
    chunks = split_documents(documents)
    print(f"{len(chunks)} szeletre vágva.")

    # 3. Betoljuk az adatbázisba
    add_chroma(chunks)
    print("Adatbázis frissítve, te kész is vagy, queen!")


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
        query_rag(query_text)
        query_text = input(": ")

# Ez biztosítja, hogy csak akkor fusson le, ha ezt a fájlt indítod el!
if __name__ == "__main__":
    main()