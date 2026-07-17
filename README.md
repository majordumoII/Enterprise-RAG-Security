## **Proj-2 Enterprise RAG Architecture with Security Guardrails**

This project takes the clean data from Project 1 and builds a secure, highly accurate Retrieval-Augmented Generation (RAG) system.

- **The Problem:** Standard AI chatbots hallucinate and leak sensitive corporate data to unauthorized users.
- **What It Builds:** A vector search engine that connects your Project 1 data to a Large Language Model (LLM). You will implement a custom security layer that filters search results based on user permission levels and blocks harmful inputs.
- **How It Connects:** This secure knowledge base serves as the brain for the autonomous agent you will build in Project 3.
- **Tech Stack:** Pinecone or Qdrant vector databases, LlamaIndex, OpenAI API or DeepSeek open-source models, and NeMo Guardrails.
