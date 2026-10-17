import ollama

# Load the dataset

dataset = []
with open('pokemon_dataset/pokemon_utf8.csv', 'r', encoding='utf-8') as file:
  dataset = file.readlines()
  print(f'Loaded {len(dataset)} entries')

# Implement the retrieval system
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# Each element in the VECTOR_DB will be a tuple (chunk, embedding)
# The embedding is a list of floats, for example: [0.1, 0.04, -0.34, 0.21, ...]
VECTOR_DB = []

def add_chunk_to_database(chunk):
  embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
  VECTOR_DB.append((chunk, embedding))

for i, chunk in enumerate(dataset):
  add_chunk_to_database(chunk)
  print(f'\rAdded chunk {i+1}/{len(dataset)} to the database', end='', flush=True)
print() 

def cosine_similarity(a, b):
  dot_product = sum([x * y for x, y in zip(a, b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([x ** 2 for x in b]) ** 0.5
  return dot_product / (norm_a * norm_b)

def retrieve(query, top_n=3):
  query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
  # temporary list to store (chunk, similarity) pairs
  similarities = []
  for chunk, embedding in VECTOR_DB:
    similarity = cosine_similarity(query_embedding, embedding)
    similarities.append((chunk, similarity))
  # sort by similarity in descending order, because higher similarity means more relevant chunks
  similarities.sort(key=lambda x: x[1], reverse=True)
  # finally, return the top N most relevant chunks
  return similarities[:top_n]

# Chatbot

# input_query = input('Ask me a question: ')
# retrieved_knowledge = retrieve(input_query)

# print('Retrieved knowledge:')
# for chunk, similarity in retrieved_knowledge:
#   print(f' - (similarity: {similarity:.2f}) {chunk}')

# instruction_prompt = f'''You are a helpful chatbot.
# Use only the following pieces of context to answer the question. Don't make up any new information:
# {'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
# '''
# # print(instruction_prompt)

# stream = ollama.chat(
#   model=LANGUAGE_MODEL,
#   messages=[
#     {'role': 'system', 'content': instruction_prompt},
#     {'role': 'user', 'content': input_query},
#   ],
#   stream=True,
# )

# # print the response from the chatbot in real-time
# print('Chatbot response:')
# for chunk in stream:
#   print(chunk['message']['content'], end='', flush=True)

# ---------------------------------------------------------------------------
# Replace everything from the "# Chatbot" line (or from the previous version of
# this section) to the END of your script with this file's contents. Keep
# everything above it (dataset loading, embedding, retrieve()) exactly as it is.
# Nothing about your database or embeddings changes, so questions_ready.json
# and the evaluation scripts still work.
#
#   python your_script.py            live side-by-side demo
#   python your_script.py 3          same, but the model sees 3 rows
#   python your_script.py score      score over the question list
#   python your_script.py score 3    score with 3 rows
# ---------------------------------------------------------------------------
import csv
import json
import sys

# How many retrieved rows the model sees. A small model is easily confused by
# extra rows, so 1 is the default. Compare with:  score 1   and   score 3
TOP_N = 1

# Keep both models loaded in memory between questions (Ollama's default is to
# unload after 5 minutes, which makes the next question slow).
KEEP_ALIVE = '30m'

# temperature 0 + fixed seed: the same question gives the same answer, so the
# examples you rehearse are the ones you get on stage. num_predict caps the
# answer length so a rambling answer can't stall the demo.
OPTIONS = {'temperature': 0, 'seed': 42, 'num_predict': 100}

# --- Faster retrieval -------------------------------------------------------
# Same cosine similarity as your original retrieve(), but computed in one
# vectorized step instead of a Python loop over 1026 x 768 numbers.
# If numpy is not installed, your original retrieve() is kept (pip install numpy).
try:
    import numpy as np

    _CHUNKS = [chunk for chunk, _embedding in VECTOR_DB]
    _EMB = np.array([embedding for _chunk, embedding in VECTOR_DB], dtype=np.float32)
    _EMB /= np.maximum(np.linalg.norm(_EMB, axis=1, keepdims=True), 1e-12)

    def _embed(text):
        try:
            return ollama.embed(model=EMBEDDING_MODEL, input=text, keep_alive=KEEP_ALIVE)['embeddings'][0]
        except TypeError:  # older ollama package without keep_alive
            return ollama.embed(model=EMBEDDING_MODEL, input=text)['embeddings'][0]

    def retrieve(query, top_n=3):
        q = np.array(_embed(query), dtype=np.float32)
        q /= max(float(np.linalg.norm(q)), 1e-12)
        sims = _EMB @ q
        best = np.argsort(-sims)[:top_n]
        return [(_CHUNKS[i], float(sims[i])) for i in best]
except ImportError:
    pass

# --- Exact-name matching (the "keyword" half of hybrid search) -------------
# Vector search on raw CSV rows is weak at picking out a Pokemon by NAME: it
# returned Crustle for a question about Squirtle. If the question contains a
# Pokemon name from your dataset, that row goes first and the vector search
# fills any remaining slots. Set NAME_MATCH = False to see the difference.
NAME_MATCH = True

import re


def _build_name_index():
    index = {}
    for chunk, _embedding in VECTOR_DB:
        try:
            fields = next(csv.reader([chunk.strip()]))
        except Exception:
            continue
        # Real rows only: 55 fields and a numeric Pokedex number. This skips the
        # header row, whose "name" column would otherwise match the word "name".
        if len(fields) < 55 or not fields[0].strip().isdigit():
            continue
        name = fields[2].strip().lower()
        if name:
            index.setdefault(name, []).append(chunk)
    return {n: (re.compile(r'(?<!\w)' + re.escape(n) + r'(?!\w)'), rows)
            for n, rows in index.items()}


_NAME_INDEX = _build_name_index()


def rows_named_in(query):
    """Rows for every Pokemon name found in the question, in the order mentioned.
    Whole-word match, so 'Mew' does not match inside 'Mewtwo'."""
    q = query.lower().replace('’', "'")
    found = []
    for name, (pattern, rows) in _NAME_INDEX.items():
        if name in q:
            m = pattern.search(q)
            if m:
                found.append((m.start(), rows))
    found.sort(key=lambda item: item[0])
    return [chunk for _position, rows in found for chunk in rows]


_vector_retrieve = retrieve  # the numpy version above, or your original one


def retrieve(query, top_n=3):
    named = rows_named_in(query) if NAME_MATCH else []
    if not named:
        return _vector_retrieve(query, top_n)
    results = [(chunk, 1.0) for chunk in named][:top_n]
    if len(results) < top_n:
        taken = {chunk for chunk, _similarity in results}
        for chunk, similarity in _vector_retrieve(query, top_n + len(results)):
            if chunk not in taken:
                results.append((chunk, similarity))
                taken.add(chunk)
            if len(results) == top_n:
                break
    return results


# (question, keywords that must ALL appear in a correct answer, lowercase)
# Answers come from rows in your own dataset. Most of these are facts a small
# model is unlikely to know from memory: Gen 8/9 Pokemon, the dataset's
# romanized Japanese names, and its "category" field.
QUESTIONS = [
    ("What is the National Pokédex number of Zarude?", ["893"]),
    ("What are the two types of Zarude?", ["dark", "grass"]),
    ("What is the National Pokédex number of Falinks?", ["870"]),
    ("What are the types of Runerigus?", ["ground", "ghost"]),
    ("What are the types of Inkay?", ["dark", "psychic"]),
    ("What type is Spoink?", ["psychic"]),
    ("What is the National Pokédex number of Yamask?", ["562"]),
    ("What is the National Pokédex number of Cramorant?", ["845"]),
    ("What are the types of Bramblin?", ["grass", "ghost"]),
    ("What is the National Pokédex number of Emolga?", ["587"]),
    ("What is the National Pokédex number of Sprigatito?", ["906"]),
    ("What are the types of Tinkatink?", ["fairy", "steel"]),
    ("What is the National Pokédex number of Finizen?", ["963"]),
    ("What are the types of Duraludon?", ["steel", "dragon"]),
    ("What is the National Pokédex number of Treecko?", ["252"]),
    ("What is the Japanese name of Falinks?", ["tairetsu"]),
    ("What is the Japanese name of Spoink?", ["baneboo"]),
    ("What is the Japanese name of Charjabug?", ["dendimushi"]),
    ("What is the category of Falinks?", ["formation"]),
    ("What is the category of Yamask?", ["spirit"]),
    ("What does Bulbasaur evolve into?", ["ivysaur"]),
    ("What does Karrablast evolve into?", ["escavalier"]),
    ("What does Squirtle evolve into?", ["wartortle"]),
]

DONT_KNOW_MARKERS = ("don't know", "do not know", "not sure", "can't find", "cannot find")


def describe(chunk):
    """Turn one raw CSV row into short labeled text that a small model can read.

    Row layout, confirmed from your printed rows: 0 number, 1 generation,
    2 name, 3 Japanese name, 4 type 1, 5 type 2, 6 category, last field =
    description. Everything else is ignored on purpose."""
    try:
        f = next(csv.reader([chunk.strip()]))
    except Exception:
        return chunk.strip()
    if len(f) < 8:
        return chunk.strip()
    types = ' and '.join(t.strip() for t in (f[4], f[5]) if t.strip())
    return (f"Name: {f[2].strip()}. Pokédex number: {f[0].strip()}. "
            f"Generation: {f[1].strip()}. Japanese name: {f[3].strip()}. "
            f"Type: {types}. Category: {f[6].strip()}. "
            f"{evolution_text(f)} "
            f"Description: {f[-1].strip()}")


def evolution_text(f):
    """Evolution line, read from the END of the row (the 5 fields that sit just
    before the 5 'other forms' fields and the description):
    stage 1, method 1, stage 2, method 2, stage 3."""
    if len(f) < 55:
        return "Evolution: not listed."
    s1, m1, s2, m2, s3 = [x.strip() for x in f[-11:-6]]
    if not s2:
        return "Evolution: none listed."
    text = f"{s1} evolves into {s2} ({m1 or 'method not given'})"
    if s3:
        text += f", and {s2} evolves into {s3} ({m2 or 'method not given'})"
    return f"Evolution: {text}."


def answer(query, use_rag, stream=False, retrieved=None):
    """Same model, same settings. The ONLY difference is whether it gets context.

    stream=True prints the answer word by word as it is generated (for the live
    demo). retrieved lets the caller pass rows it already looked up."""
    if use_rag:
        if retrieved is None:
            retrieved = retrieve(query, top_n=TOP_N)
        facts = '\n'.join(f'- {describe(chunk)}' for chunk, _similarity in retrieved)
        system = (
            "You answer questions about Pokémon using only the facts below.\n"
            "If the facts do not contain the answer, say you don't know.\n"
            "Answer in one short sentence.\n\n"
            f"Facts:\n{facts}"
        )
    else:
        retrieved = []
        system = "You are a helpful chatbot. Answer in one short sentence."
    request = dict(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': query},
        ],
        options=OPTIONS,
        keep_alive=KEEP_ALIVE,
    )
    if stream:
        text = ''
        for part in ollama.chat(stream=True, **request):
            piece = part['message']['content']
            print(piece, end='', flush=True)
            text += piece
        print()
    else:
        text = ollama.chat(**request)['message']['content']
    return text, retrieved


def said_dont_know(text):
    t = text.lower().replace('’', "'")
    return any(m in t for m in DONT_KNOW_MARKERS)


def compare(query):
    retrieved = retrieve(query, top_n=TOP_N)  # a few milliseconds
    print(f'\nQUESTION: {query}')
    print('\n--- WITHOUT RAG (the model\'s memory only) ---')
    answer(query, use_rag=False, stream=True)
    print('\n--- WITH RAG ---')
    for chunk, similarity in retrieved:
        print(f'  retrieved ({similarity:.2f}): {describe(chunk)[:110]}...')
    answer(query, use_rag=True, stream=True, retrieved=retrieved)


def run_scoreboard():
    results = []
    for question, keywords in QUESTIONS:
        plain, _ = answer(question, use_rag=False)
        rag, _ = answer(question, use_rag=True)
        ok_plain = all(k.lower() in plain.lower() for k in keywords)
        ok_rag = all(k.lower() in rag.lower() for k in keywords)
        abstained = said_dont_know(rag)
        results.append({
            "question": question, "keywords": keywords,
            "plain_correct": ok_plain, "rag_correct": ok_rag,
            "rag_said_dont_know": abstained,
            "plain_answer": plain, "rag_answer": rag,
        })
        print(f'no-RAG {"PASS" if ok_plain else "FAIL"} | RAG {"PASS" if ok_rag else "FAIL"}'
              f'{" (said it did not know)" if abstained else ""} | {question}')
    n = len(results)
    print(f'\nRows given to the model (TOP_N): {TOP_N}')
    print(f'Without RAG: {sum(r["plain_correct"] for r in results)}/{n}'
          f'   With RAG: {sum(r["rag_correct"] for r in results)}/{n}')
    print(f'RAG said it did not know on {sum(r["rag_said_dont_know"] for r in results)} of {n}')
    with open('scoreboard.json', 'w', encoding='utf8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('Full answers saved to scoreboard.json. Read them: the check is only keyword matching.')


if __name__ == '__main__':
    args = sys.argv[1:]
    numbers = [a for a in args if a.isdigit()]
    if numbers:
        TOP_N = int(numbers[0])
    if 'score' in args:
        run_scoreboard()
    else:
        print(f'(The RAG side sees the top {TOP_N} retrieved row(s).)')
        while True:
            q = input('\nAsk me a question (press Enter to quit): ').strip()
            if not q:
                break
            compare(q)