#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from jogi_agent.crew import JogiAgent

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    question = input(": ")

    while question != "break":
        inputs = {
            'topic': question,
            'current_year': datetime.now().year
        }

        try:
            result = JogiAgent().crew().kickoff(inputs=inputs)
            print(result.raw)
        except Exception as e:
            raise Exception(f"An error occurred while running the crew: {e}")

        question = input(": ")

def test_questions():
    """
        Run the crew on test questions.
    """
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

        inputs = {
            'topic': question,
            'current_year': datetime.now().year
        }

        try:
            result = JogiAgent().crew().kickoff(inputs=inputs)
            print(result.raw)
        except Exception as e:
            raise Exception(f"An error occurred while running the crew: {e}")



def train():
    """
    Train the crew for a given number of iterations.
    """
    question = input(": ")

    inputs = {
        'topic': question,
        'current_year': datetime.now().year
    }
    try:
        JogiAgent().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        JogiAgent().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    question = input(": ")

    inputs = {
        'topic': question,
        'current_year': datetime.now().year
    }

    try:
        JogiAgent().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

def run_with_trigger():
    """
    Run the crew with trigger payload.
    """
    import json

    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    inputs = {
        "crewai_trigger_payload": trigger_payload,
        "topic": "",
        "current_year": ""
    }

    try:
        result = JogiAgent().crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")

if __name__ == "__main__":
    test_questions()