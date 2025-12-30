import ollama

# Load dataset
dataset = []
with open("cat_facts.txt", "r") as f:
    dataset = f.readlines()
    print(f"Loaded {len(dataset)} log entries.")

# Ollama models
embedding_model = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf:latest"
language_model = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest"

# make a vector store

vector_store = []


def add_chuck_to_vdb(chunk):
    embeddings = ollama.embed(model=embedding_model, input=chunk)["embeddings"][0]
    vector_store.append((chunk, embeddings))


# * assuming each line a chunk
for i, chunk in enumerate(dataset):
    add_chuck_to_vdb(chunk)
    if i % 100 == 0:
        print(f"Added {i} chunks to vector store.")


# Define similarity score function
def cosine_similarity(x, y):
    dot_product = sum([a * b for a, b in zip(x, y)])
    norm_a = sum([a**2 for a in x]) ** 0.5
    norm_b = sum([b**2 for b in y]) ** 0.5
    return dot_product / (norm_a * norm_b)


# Function to retrieve relevant chunks
def retriever(query, top_k=3):
    query_embeddding = ollama.embed(model=embedding_model, input=query)["embeddings"][0]
    similarities = []

    for chunk, embedding in vector_store:
        score = cosine_similarity(query_embeddding, embedding)
        similarities.append((chunk, score))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


# generation part
input_query = input("Ask me a question:\n")
retrieved_knowledge = retriever(input_query, top_k=5)

print("Retrieved knowledge:")
for chunk, similarity in retrieved_knowledge:
    print(f" - (similarity: {similarity:.2f}) {chunk}")

instruction_prompt = f"""You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{"\n".join([f" - {chunk}" for chunk, similarity in retrieved_knowledge])}
"""

# response
stream = ollama.chat(
    model=language_model,
    messages=[
        {"role": "system", "content": instruction_prompt},
        {"role": "user", "content": input_query},
    ],
    stream=True,
)

print("Response:")
for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)
print()  # for newline after response
