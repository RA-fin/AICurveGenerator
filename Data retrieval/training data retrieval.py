# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 10:24:50 2025

@author: ...
"""

import requests
from bs4 import BeautifulSoup
import os
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
from transformers import pipeline ,AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util

import re
import unicodedata


BASE_URL = "https://www.snb.ch/en/news-publications/economy/quarterly-bulletin/quarterly-bulletin~page-1="
OUTPUT_DIR = "snb_reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_all_links(url,CSSclass):
    resp = requests.get(url,verify=False)
    soup = BeautifulSoup(resp.text, "html.parser",)
    links = [a["href"] for a in soup.find_all("a", class_=CSSclass) if a.has_attr("href")]
    return list(set(links))

def download_pdf(url):
    filename=r"C:\temp\report.pdf"
    r = requests.get(url,verify=False)
    with open(filename, "wb") as f:
        f.write(r.content)
    return filename

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    text=re.sub(r'[^\x09\x0A\x0D\x20-\x7E]', '', text)
    return text

def chunk_text(text, chunk_size=1000): 
    words = text.split()
    chunks=[]
    i=0
    while i<len(words):
        l=0
        thisChunk=""
        while (l<chunk_size) &  (i<len(words)):
            thisChunk=thisChunk+" "+words[i]
            i+=1
            l=len(thisChunk)
            #print(thisChunk)
        chunks=chunks+[thisChunk]
            
    return chunks

def truncate_text(text, model_name, max_length=512):
    """
    Truncate text to fit transformer model limits
    
    Args:
        text (str): Input text to truncate
        model_name (str): Transformer model identifier
        max_length (int): Maximum sequence length
    
    Returns:
        dict: Tokenized and truncated text
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize with truncation enabled
    tokens = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    
    return tokenizer.decode(tokens["input_ids"][0])

def filter_economic_chunks(chunks):
    results = []
    print("Filtering economic relevant parts...")
    for i in range(len(chunks)):
        result = classifier(chunks[i], candidate_labels=["economy switzerland", "other"])
        if result['labels'][0] == "economy switzerland": #and result['scores'][0] > 0.9:
            results.append(result['scores'][0])
        else:
            results.append(1-result['scores'][0])
            #relevant_chunks.append(chunks[i])
        print("\rProgress: "+str(round(100*(i+1)/len(chunks),1))+"%", end="", flush=True)
    relevant_chunks=[chunks[i] for i in range(len(results)) if results[i]>np.quantile(results, 0.975)]
    print()
    return relevant_chunks

def summarize_text(text):
    print("Summarizing...")
    summary = summarizer(text, max_length=200, min_length=50, do_sample=False,truncation=True)
    return summary[0]['summary_text']

def remove_redundancy_semantic(summary, similarity_threshold=0.8):
    sentences = [s.strip() for s in summary.split(". ") if s.strip()]
    embeddings = embedder.encode(sentences, convert_to_tensor=True)
    unique_sentences = []
    for i, sentence in enumerate(sentences):
        if all(util.cos_sim(embeddings[i], embeddings[j]) < similarity_threshold for j in range(len(unique_sentences))):
            unique_sentences.append(sentence)
    return ". ".join(unique_sentences)

pageLinks = []
for page in range(1, 11):  # Pages 1-10
    url = f"{BASE_URL}{page}~"
    pageLinks=pageLinks+get_all_links(url,"link-teaser h-typo-body")
    
pdfLinks = []
for page in pageLinks:  # Pages 1-10
    url = "https://www.snb.ch"+page
    pdfLinks=pdfLinks+get_all_links(url,"m-mixed-list-item h-typo-body")

#Get the economic scenarios from SNB bullettins

# Load local models
classifier = pipeline("zero-shot-classification", model=r"C:\Users\...\Documents\models\valhalla\distilbart-mnli-12-9")
summarizer = pipeline("summarization", model=r"C:\Users\...\Documents\models\bart-base\bart-base")
embedder = SentenceTransformer(r"C:\Users\...\Documents\models\allMiniLM\all-MiniLM-L6-v2")  # Lightweight Sentence-BERT model

narratives=[]
for thispdf in pdfLinks:
    print(thispdf)
    download_pdf("https://www.snb.ch"+thispdf)
    raw_text = extract_text_from_pdf(r"C:\temp\report.pdf")
    chunks = (chunk_text(raw_text))
    economic_chunks = filter_economic_chunks(chunks)

    if economic_chunks:
        combined_text = " ".join(economic_chunks)
        combined_text = remove_redundancy_semantic(combined_text)
        combined_text = truncate_text(combined_text,r"C:\Users\...\Documents\models\valhalla\distilbart-mnli-12-9")
        summary = summarize_text(combined_text)
        clean_summary = remove_redundancy_semantic(summary)
        
    narratives.append(clean_summary)

dates=[re.findall("[0-9]{4}_[0-9]{1}",x)[0] for x in pdfLinks]
dates=[x[0:4]+"-"+str(int(x[-1])*3)+"-"+"15" for x in dates]

historicScenarios=pd.DataFrame()
historicScenarios["Date"]=dates
historicScenarios["Narratives"]=narratives

historicScenarios.to_excel(r"C:\Users\...\...\Documents\Personal scripts\Scenaio generation\scenarios.xlsx")

NSparams=pd.read_csv(r"C:\Users\...\...\Documents\Personal scripts\Scenaio generation\snb-data-rendopar.csv",
                     skiprows=3,sep=";")
NSparams=NSparams.pivot(columns=["d0"],values="Value",index="Date")
NSparams.dropna(inplace=True)

historicScenarios["date"]=[pd.Timestamp(x) for x in historicScenarios["Date"]]
historicScenarios.sort_values("date",inplace=True)
NSparams["date"]=[pd.Timestamp(x) for x in NSparams.index]
historicScenarios=pd.merge_asof(historicScenarios,NSparams,on="date",direction="nearest")

historicScenarios.to_excel(r"C:\Users\...\...\Documents\Personal scripts\Scenaio generation\scenarios.xlsx")
