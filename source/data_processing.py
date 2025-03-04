# src/data_processing.py
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# 원본 PDF 폴더 경로
RAW_PDF_DIR = "../data/raw_pdfs"
OUTPUT_FILE = "../data/processed/corpus.txt"

def load_and_split_pdfs(pdf_dir=RAW_PDF_DIR):
    text_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_dir, filename)
            loader = PyPDFLoader(file_path)
            documents = loader.load()  # PDF -> Document 객체 리스트
            # 각 Document의 page_content에서 텍스트 추출 및 분할
            docs = splitter.split_documents(documents)
            for doc in docs:
                text = doc.page_content.strip()
                if text:
                    text_chunks.append(text)
    return text_chunks

if __name__ == "__main__":
    chunks = load_and_split_pdfs()
    # 추출된 텍스트 청크들을 파일로 저장
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.replace("\n", " ") + "\n")  # 청크를 한 줄로 저장
    print(f"Processed {len(chunks)} text chunks and saved to {OUTPUT_FILE}")
