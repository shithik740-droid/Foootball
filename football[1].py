import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


# ==================================================
# LOAD ENVIRONMENT VARIABLES
# ==================================================

load_dotenv()

# Get OpenRouter API key
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    print("ERROR: OPENROUTER_API_KEY not found.")
    print("Please check your .env file.")
    exit()


# ==================================================
# CREATE OPENROUTER CLIENT
# ==================================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)


# ==================================================
# 1. LOAD FOOTBALL DATA
# ==================================================

def load_data():

    with open("football.txt", "r", encoding="utf-8") as file:
        text = file.read()

    return text


# ==================================================
# 2. CHUNK THE DATA
# ==================================================

def create_chunks(text):

    chunks = []

    # Split text using blank lines
    paragraphs = text.split("\n\n")

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if paragraph:
            chunks.append(paragraph)

    return chunks


# ==================================================
# 3. CREATE EMBEDDING
# ==================================================

def create_embedding(text):

    response = client.embeddings.create(

        model="openai/text-embedding-3-small",

        input=text

    )

    return response.data[0].embedding


# ==================================================
# 4. CREATE VECTOR DATABASE
# ==================================================

def create_vector_database(chunks):

    vectors = []

    print("\nCreating embeddings...")

    for i, chunk in enumerate(chunks):

        print(
            f"Embedding chunk {i + 1}/{len(chunks)}"
        )

        embedding = create_embedding(chunk)

        vectors.append(embedding)

    return np.array(vectors)


# ==================================================
# 5. COSINE SIMILARITY
# ==================================================

def cosine_similarity(a, b):

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0

    return np.dot(a, b) / denominator


# ==================================================
# 6. RETRIEVE RELEVANT INFORMATION
# ==================================================

def retrieve_information(question, chunks, vectors):

    print("\nSearching knowledge base...")

    # Create embedding for question
    question_embedding = create_embedding(question)

    similarities = []

    for vector in vectors:

        similarity = cosine_similarity(
            question_embedding,
            vector
        )

        similarities.append(similarity)

    # Find most relevant chunk
    best_index = np.argmax(similarities)

    best_score = similarities[best_index]

    print(
        f"Best similarity score: {best_score:.4f}"
    )

    return chunks[best_index]


# ==================================================
# 7. GENERATE ANSWER
# ==================================================

def generate_answer(question, context):

    prompt = f"""
You are a football knowledge assistant.

Answer the user's question using ONLY the
information provided in the context below.

Do NOT use outside knowledge.

If the answer is not available in the context,
say exactly:

"I don't have that information in my knowledge base."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = client.chat.completions.create(

        # OpenRouter model
        model="openai/gpt-4o-mini",

        # Limit output tokens
        max_tokens=300,

        messages=[

            {
                "role": "system",
                "content": (
                    "You answer questions using only "
                    "the provided football knowledge base."
                )
            },

            {
                "role": "user",
                "content": prompt
            }

        ]

    )

    return response.choices[0].message.content


# ==================================================
# 8. MAIN RAG PIPELINE
# ==================================================

def main():

    print("===================================")
    print("       FOOTBALL RAG SYSTEM")
    print("===================================")

    # ----------------------------------------------
    # LOAD DATA
    # ----------------------------------------------

    print("\nLoading football data...")

    text = load_data()

    if not text.strip():

        print("ERROR: football.txt is empty.")

        return

    # ----------------------------------------------
    # CREATE CHUNKS
    # ----------------------------------------------

    chunks = create_chunks(text)

    print(
        f"Loaded {len(chunks)} chunks."
    )

    # ----------------------------------------------
    # CREATE VECTOR DATABASE
    # ----------------------------------------------

    vectors = create_vector_database(chunks)

    print("\nVector database created.")

    # ----------------------------------------------
    # READY
    # ----------------------------------------------

    print("\n===================================")
    print("       RAG SYSTEM IS READY")
    print("===================================")

    print("Type 'exit' to stop.\n")

    # ----------------------------------------------
    # QUESTION LOOP
    # ----------------------------------------------

    while True:

        question = input(
            "Ask a football question: "
        )

        # Remove extra spaces
        question = question.strip()

        # Exit
        if question.lower() == "exit":

            print("\nGoodbye!")

            break

        # Empty question
        if not question:

            print("Please enter a question.\n")

            continue

        # ------------------------------------------
        # RETRIEVE INFORMATION
        # ------------------------------------------

        context = retrieve_information(
            question,
            chunks,
            vectors
        )

        # ------------------------------------------
        # GENERATE ANSWER
        # ------------------------------------------

        answer = generate_answer(
            question,
            context
        )

        # ------------------------------------------
        # DISPLAY ANSWER
        # ------------------------------------------

        print("\nAnswer:")
        print(answer)

        print("\n" + "-" * 50 + "\n")


# ==================================================
# START PROGRAM
# ==================================================

if __name__ == "__main__":

    main()