"""
Ultimate RAG: Multi-modal Retrieval-Augmented Generation System

A deterministic, multi-modal RAG system for technical documents
supporting Physics, Math, Chemistry, and Data Science content.
"""

__version__ = "1.0.0"
__author__ = "Ultimate RAG Team"

from ultimate_rag.pipeline import UltimateRAG
from ultimate_rag.config import Config

__all__ = ["UltimateRAG", "Config"]
