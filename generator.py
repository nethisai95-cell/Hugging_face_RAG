
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from retrieval import search_documents

# ==========================================
# 1. MODEL CONFIGURATION
# ==========================================

MODEL_NAME = "google/flan-t5-base"

print("Loading Hugging Face language model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)

model.eval()

print("Language model loaded successfully!")


# ==========================================
# 2. ANSWER GENERATION
# ==========================================

def generate_answer(question):

    print("\nSearching relevant documents...")

    # Retrieve relevant chunks from ChromaDB
    results = search_documents(
        question,
        top_k=3
    )

    if not results:
        return (
            "No relevant information found in the documents.",
            []
        )

    # ==========================================
    # 3. PREPARE DOCUMENT CONTEXT
    # ==========================================

    context_parts = []

    for item in results:

        context_parts.append(
            f"""
            Page: {item['page']}
            Content: {item['text']}
            """
        )

    context = "\n".join(context_parts)

    # ==========================================
    # 4. CONSTRUCT PROMPT
    # ==========================================

    prompt = f"""
    You are an intelligent document analysis assistant.

    Read the document context carefully and answer
    the user's question using only the information
    provided in the context.

    DOCUMENT CONTEXT:
    {context}

    USER QUESTION:
    {question}

    INSTRUCTIONS:
    1. Give a clear and detailed answer.
    2. Include all relevant technical skills,
       technologies and experience.
    3. Use bullet points where appropriate.
    4. Do not invent information.
    5. If the answer is not available in the
       context, clearly say so.

    ANSWER:
    """

    # ==========================================
    # 5. TOKENIZATION
    # ==========================================

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    # ==========================================
    # 6. GENERATE ANSWER
    # ==========================================

    print("Generating answer...\n")

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            num_beams=4,
            do_sample=False,
            repetition_penalty=1.2,
            early_stopping=True
        )

    # ==========================================
    # 7. DECODE ANSWER
    # ==========================================

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer, results


# ==========================================
# 8. DISPLAY ANSWER AND SOURCES
# ==========================================

def main():

    print("\n" + "=" * 55)
    print("       DOCUMENT ANALYSIS RAG SYSTEM")
    print("=" * 55)

    question = input("\nAsk a question about your PDF: ")

    if not question.strip():
        print("Please enter a valid question.")
        return

    answer, sources = generate_answer(question)

    print("\n" + "=" * 55)
    print("GENERATED ANSWER")
    print("=" * 55)

    print(answer)

    print("\n" + "=" * 55)
    print("SOURCE REFERENCES")
    print("=" * 55)

    for source in sources:

        print(
            f"Source: {source['source']} | "
            f"Page: {source['page']}"
        )

    print("\n" + "=" * 55)
    print("END OF RESPONSE")
    print("=" * 55)


# ==========================================
# 9. APPLICATION ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()
