import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from PyPDF2 import PdfReader
import torch


# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1

# Load tokenizer and model for question answering
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=device)


prompts_responses = {}


def extract_prompts_responses(text):
    # Function to extract prompts (questions) and responses from text
    prompts = []
    responses = []
    sentences = text.split("\n\n")  # Paragraph separated by double newline

    for i in range(len(sentences)):
        prompt = sentences[i].strip()
        if len(prompt) > 0:
            prompts.append(prompt)
            if i + 1 < len(sentences):
                responses.append(sentences[i + 1].strip())
            else:
                responses.append("")  # Handle cases where no subsequent sentences

    # Populate prompts_responses with extracted pairs
    for prompt, response in zip(prompts, responses):
        prompts_responses[prompt] = response


def generate_response(question):
    # Function to generate a response based on a given question using QA pipeline
    context = " ".join(prompts_responses.keys())
    result = qa_pipeline(question=question, context=context, max_answer_len=100)
    return result['answer']


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                # Replace multiple newlines with a single space
                page_text = page_text.replace('\n', ' ').replace('\r', ' ')
                text += page_text
    return text


# Streamlit app
def main():
    st.title("Company Information Chatbot")

    pdf_path = 'Corpus.pdf'  # Replace with your actual PDF path
    text = extract_text_from_pdf(pdf_path)
    extract_prompts_responses(text)

    st.markdown("### Chatbot")

    user_input = st.text_input("You:")

    if user_input:
        response = generate_response(user_input)
        st.text_area("Bot:", value=response, height=200)


if __name__ == "__main__":
    main()
