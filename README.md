
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

Developed as an AI-powered Placement Intelligence System for campus recruitment preparation.
