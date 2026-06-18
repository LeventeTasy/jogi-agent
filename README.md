# JogiAgent Crew

Welcome to the JogiAgent Crew project, powered by [crewAI](https://crewai.com). This template is designed to help you set up a multi-agent AI system with ease, leveraging the powerful and flexible framework provided by crewAI. Our goal is to enable your agents to collaborate effectively on complex tasks, maximizing their collective intelligence and capabilities.

## Installation

Ensure you have Python >=3.10 <3.14 installed on your system. This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.

First, if you haven't already, install uv:

```bash
pip install uv
```

Next, navigate to your project directory and install the dependencies:

(Optional) Lock the dependencies and install them by using the CLI command:
```bash
crewai install
```

## Running the Project

To kickstart your crew of AI agents and begin task execution, run this from the root folder of your project:

```bash
$ crewai run
```

This example, unmodified, will run the create a `report.md` file with the output of a research on LLMs in the root folder.

Multi-Agent RAG Architektúra Alkalmazása Magyar Jogi DokumentumokonProjektleírásEz a projekt egy autonóm ágensorientált megközelítésre épülő, nagydimenziós szemantikus kereső és dokumentum-elemző rendszer (Retrieval-Augmented Generation). A fejlesztés elsődleges célja komplex, strukturálatlan magyar jogi forrásszövegek – kiemelten a Polgári Törvénykönyv (PTK), a Személyi Jövedelemadó törvény (SZJA), valamint a GDPR szabályozás – hatékony feldolgozása, indexelése és kontextus-tudatos, magas pontosságú megválaszolása.  Rendszerarchitektúra és Főbb KomponensekTöbbágenses koordináció (CrewAI keretrendszer): A válaszadási folyamat elosztott intelligenciára épül. A rendszerben dedikált, autonóm ágensek (Kutató, Elemző, Kritikus) működnek együtt, amelyek feladat-delegálási folyamatokon keresztül ellenőrzik egymás kimenetét a jogi pontosság biztosítása érdekében.  Vektoros adatbázis és szemantikus keresés (ChromaDB): A jogi forrásdokumentumok beágyazását (embedding) követően a szövegrészletek lokális ChromaDB vektor-térbe kerülnek. A releváns kontextus kinyerése koszinusz-hasonlósági metrikák alapján történik.  Jogi szövegekre optimalizált chunking pipeline: Egyedi, reguláris kifejezésekre épülő darabolási stratégia, amely illeszkedik a magyar jogszabályok szerkezetéhez (cikkelyek, bekezdések, pontok). Ez biztosítja, hogy a kinyert kontextus szemantikailag egységes maradjon.  
Projektstruktúra

jogi-agent/
├── chroma_db/               # A beágyazott jogi szövegek lokális vektoros adatbázisa
├── knowledge/               # Lokális tudásbázis elemek (user_preference.txt)
├── pdf/                     # A feldolgozott forrásdokumentumok (PTK, SZJA, GDPR)
├── src/                     # A forráskódot tartalmazó főkönyvtár
│   └── jogi_agent/          
│       ├── config/          # Az ágensek és feladatok YAML/konfigurációs fájljai
│       ├── tools/           # Egyedi ágens-eszközök
│       │   ├── init.py
│       │   └── custom_tool.py
│       ├── init.py
│       ├── crew.py          # Az ágensek és feladatok logikai összekapcsolása
│       ├── main.py          # **A Crew futtatásáért felelős belépési pont**
│       └── report.md        # Lokális jelentés
│   └── rag.py               # A RAG pipeline és a ChromaDB lekérdezések implementációja
├── .env.example             # Környezeti változók sablonja az API integrációhoz
├── .gitignore               # Verziókezelésből kizárt fájlok listája
├── pyproject.toml           # Projekt metaadatok és függőségek definíciója
├── README.md                # Projekt szintű fő dokumentáció
└── uv.lock                  # Az uv dependency manager zárolási fájlja  Telepítés és Futtatás1. Környezet előkészítéseA rendszer futtatásához Python 3.10+ környezet szükséges. A függőségek kezelése a modern és gyors uv csomagkezelővel történik. Hozzon létre egy virtuális környezetet, majd szinkronizálja a csomagokat:  uv venvsource .venv/bin/activateuv pip compile requirements.txt -o requirements.txtuv pip sync2. Környezeti változók konfigurálásaA projekt a biztonsági előírásoknak megfelelően nem tartalmaz beágyazott API kulcsokat. A futtatáshoz szükséges a környezeti változók beállítása a lokális fájlban.  Másolja le a mintafájlt az alábbi paranccsal:cp .env.example .envEzt követően a létrejött .env fájlban adja meg a releváns hozzáférési kulcsokat:OPENAI_API_KEY=az_on_openai_kulcsaCREWAI_API_KEY=az_on_crewai_kulcsaCHROMA_DB_PATH=./chroma_db3. A pipeline indításaA teljes munkafolyamat futtatása a gyökérkönyvtárban található bin.py szkript meghívásával történik:python bin.py
