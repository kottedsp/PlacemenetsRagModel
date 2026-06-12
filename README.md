# PlacemenetsRagModel
# Placement Intelligence Portal

## Overview

An AI-powered web application that helps students prepare for campus placements by providing company insights, eligibility checking, placement statistics, and an intelligent RAG-based chatbot.

## Features

* AI Placement Advisor (RAG Chatbot)
* Eligibility Checker (CGPA & Backlog Based)
* Company Explorer
* Placement Analytics Dashboard
* Semantic Search using FAISS

## Tech Stack

**Frontend:** HTML, CSS, JavaScript
**Backend:** Flask, Python
**AI/RAG:** LangChain, FAISS, HuggingFace Embeddings, Ollama (Llama 3)

## How It Works

1. User asks a placement-related question.
2. FAISS retrieves relevant company records.
3. Llama 3 generates answers using retrieved data.
4. Results are displayed with source companies.

## Run the Project

### Install Dependencies

```bash
pip install flask flask-cors langchain langchain-community
pip install langchain-huggingface sentence-transformers
pip install faiss-cpu langchain-ollama
```

### Start Ollama

```bash
ollama pull llama3
```

### Run Backend

```bash
python rag_api.py
```

### Launch Frontend

Open `index.html` in a browser.

## Key APIs

* `POST /api/ask` – AI chatbot
* `POST /api/eligibility` – Eligibility checker
* `GET /api/companies` – Company data
* `GET /api/stats` – Placement statistics
* `GET /health` – Health check

## Author

Developed as an AI-powered Placement Intelligence System for campus recruitment preparation.
