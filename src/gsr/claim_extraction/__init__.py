"""Claim extraction – Module 2.

Decompose reviews into verifiable factual claims via LLM-based extraction.
"""

from gsr.claim_extraction.extractor import extract_all_claims, extract_review_claims
from gsr.claim_extraction.storage import init_claims_db, save_extraction_results
