"""
BioPortal Medical Entity Verification.

Uses the NCBO BioPortal API to verify that medical entities mentioned in the
model's answer are real, recognized medical terms. This helps catch hallucinated
drug names, disease names, procedures, etc.

BioPortal covers 1000+ biomedical ontologies including:
- SNOMED CT, MeSH, ICD-10, RxNorm, NCI Thesaurus, LOINC, etc.

Requires a BioPortal API key: https://bioportal.bioontology.org/account
"""

import logging
import re
import time
from functools import lru_cache
from typing import Optional

import requests

logger = logging.getLogger(__name__)

BIOPORTAL_SEARCH_URL = "https://data.bioontology.org/search"
BIOPORTAL_ANNOTATOR_URL = "https://data.bioontology.org/annotator"

# Key ontologies for medical entity verification
DEFAULT_ONTOLOGIES = [
    "SNOMEDCT",  # SNOMED Clinical Terms
    "MESH",       # Medical Subject Headings
    "ICD10CM",    # ICD-10 Clinical Modification
    "RXNORM",     # Drug vocabulary
    "NCIT",       # NCI Thesaurus
    "LOINC",      # Lab tests
    "DOID",       # Disease Ontology
    "HP",         # Human Phenotype Ontology
]


class BioPortalChecker:
    """
    Verifies medical entities against BioPortal ontologies.

    Usage:
        checker = BioPortalChecker(api_key="your-api-key")
        result = checker.verify_entity("metformin")
        # result = {"found": True, "ontologies": ["RXNORM", "MESH"], "score": 1.0}
    """

    def __init__(
        self,
        api_key: str = "",
        ontologies: list[str] | None = None,
        cache_size: int = 10000,
        timeout: int = 10,
        rate_limit_delay: float = 0.1,
    ):
        """
        Args:
            api_key: BioPortal API key
            ontologies: List of ontology acronyms to search
            cache_size: LRU cache size for entity lookups
            timeout: API request timeout in seconds
            rate_limit_delay: Delay between API calls (BioPortal has rate limits)
        """
        self.api_key = api_key
        self.ontologies = ontologies or DEFAULT_ONTOLOGIES
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self._cache = {}
        self._cache_size = cache_size
        self._last_request_time = 0

    def _rate_limit(self):
        """Simple rate limiter."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def verify_entity(self, entity: str) -> dict:
        """
        Verify a single medical entity against BioPortal.

        Args:
            entity: Medical entity string (e.g., "metformin", "diabetes mellitus")

        Returns:
            dict with:
                - found: bool — whether entity was found in any ontology
                - ontologies: list[str] — which ontologies matched
                - preferred_label: str — canonical name if found
                - score: float — confidence score [0, 1]
        """
        entity_lower = entity.lower().strip()

        # Check cache
        if entity_lower in self._cache:
            return self._cache[entity_lower]

        # Skip very short or generic terms
        if len(entity_lower) < 2 or entity_lower in _STOP_WORDS:
            result = {"found": False, "ontologies": [], "preferred_label": "", "score": 0.0}
            self._cache_entity(entity_lower, result)
            return result

        if not self.api_key:
            # No API key — use heuristic matching
            return self._heuristic_verify(entity_lower)

        # Query BioPortal Search API
        result = self._search_bioportal(entity_lower)
        self._cache_entity(entity_lower, result)
        return result

    def verify_entities(self, entities: list[str]) -> dict:
        """
        Verify multiple entities and return aggregate score.

        Returns:
            dict with:
                - entity_results: list of individual results
                - verified_count: number of verified entities
                - total_count: total entities checked
                - verification_score: float [0, 1]
        """
        if not entities:
            return {
                "entity_results": [],
                "verified_count": 0,
                "total_count": 0,
                "verification_score": 0.0,
            }

        results = []
        verified = 0
        for entity in entities:
            result = self.verify_entity(entity)
            results.append({"entity": entity, **result})
            if result["found"]:
                verified += 1

        total = len(entities)
        return {
            "entity_results": results,
            "verified_count": verified,
            "total_count": total,
            "verification_score": verified / total if total > 0 else 0.0,
        }

    def annotate_text(self, text: str) -> dict:
        """
        Use BioPortal Annotator to find all medical entities in text.
        This is an alternative to LLM-based entity extraction.

        Returns:
            dict with entities found and their ontology sources
        """
        if not self.api_key or not text:
            return {"entities": [], "count": 0}

        self._rate_limit()

        params = {
            "apikey": self.api_key,
            "text": text[:5000],
            "ontologies": ",".join(self.ontologies),
            "longest_only": "true",
            "exclude_numbers": "true",
            "minimum_match_length": 3,
        }

        try:
            response = requests.get(
                BIOPORTAL_ANNOTATOR_URL,
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            annotations = response.json()

            entities = []
            seen = set()
            for ann in annotations:
                cls = ann.get("annotatedClass", {})
                label = cls.get("prefLabel", "")
                if label and label.lower() not in seen:
                    seen.add(label.lower())
                    ontology = cls.get("links", {}).get("ontology", "").split("/")[-1]
                    entities.append({
                        "label": label,
                        "ontology": ontology,
                        "match_type": ann.get("annotations", [{}])[0].get("matchType", ""),
                    })

            return {"entities": entities, "count": len(entities)}

        except Exception as e:
            logger.warning(f"BioPortal annotator error: {e}")
            return {"entities": [], "count": 0}

    def _search_bioportal(self, entity: str) -> dict:
        """Query BioPortal search API."""
        self._rate_limit()

        params = {
            "apikey": self.api_key,
            "q": entity,
            "ontologies": ",".join(self.ontologies),
            "require_exact_match": "false",
            "pagesize": 5,
        }

        try:
            response = requests.get(
                BIOPORTAL_SEARCH_URL,
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            results = data.get("collection", [])
            if results:
                top = results[0]
                matched_ontologies = list(set(
                    r.get("links", {}).get("ontology", "").split("/")[-1]
                    for r in results[:5]
                ))
                preferred_label = top.get("prefLabel", entity)

                # Score based on match quality
                exact_match = preferred_label.lower() == entity.lower()
                score = 1.0 if exact_match else 0.7

                return {
                    "found": True,
                    "ontologies": matched_ontologies,
                    "preferred_label": preferred_label,
                    "score": score,
                }
            else:
                return {
                    "found": False,
                    "ontologies": [],
                    "preferred_label": "",
                    "score": 0.0,
                }

        except Exception as e:
            logger.warning(f"BioPortal search error for '{entity}': {e}")
            return {
                "found": False,
                "ontologies": [],
                "preferred_label": "",
                "score": 0.0,
            }

    def _heuristic_verify(self, entity: str) -> dict:
        """
        Fallback heuristic when no API key is available.
        Uses basic pattern matching for common medical terms.
        """
        # Check against common medical suffixes/patterns
        medical_patterns = [
            r".*itis$",       # inflammation (arthritis, bronchitis)
            r".*osis$",       # condition (fibrosis, stenosis)
            r".*emia$",       # blood condition (anemia, leukemia)
            r".*oma$",        # tumor (carcinoma, melanoma)
            r".*pathy$",      # disease (neuropathy, myopathy)
            r".*ectomy$",     # surgical removal (appendectomy)
            r".*plasty$",     # surgical repair (angioplasty)
            r".*scopy$",      # examination (endoscopy)
            r".*ology$",      # study of (cardiology)
            r".*pril$",       # ACE inhibitors (lisinopril)
            r".*sartan$",     # ARBs (losartan, valsartan)
            r".*statin$",     # statins (atorvastatin)
            r".*olol$",       # beta blockers (metoprolol)
            r".*azole$",      # antifungals (fluconazole)
            r".*cillin$",     # penicillins (amoxicillin)
            r".*mycin$",      # macrolides (azithromycin)
            r".*formin$",     # biguanides (metformin)
            r".*gliptin$",    # DPP-4 inhibitors (sitagliptin)
            r".*mab$",        # monoclonal antibodies (trastuzumab)
            r".*nib$",        # kinase inhibitors (imatinib)
        ]

        for pattern in medical_patterns:
            if re.match(pattern, entity, re.IGNORECASE):
                return {
                    "found": True,
                    "ontologies": ["heuristic"],
                    "preferred_label": entity,
                    "score": 0.5,  # Lower confidence for heuristic
                }

        return {
            "found": False,
            "ontologies": [],
            "preferred_label": "",
            "score": 0.0,
        }

    def _cache_entity(self, key: str, value: dict):
        """Add to cache with size limit."""
        if len(self._cache) >= self._cache_size:
            # Remove oldest entry (simple FIFO, not LRU)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = value


# Common stop words to skip during entity verification
_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "shall", "must",
    "and", "or", "but", "not", "no", "yes", "it", "its",
    "this", "that", "these", "those", "i", "you", "he", "she",
    "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "his", "our", "their", "what", "which", "who", "whom", "how",
    "when", "where", "why", "if", "then", "than", "both", "each",
    "all", "any", "few", "more", "most", "other", "some", "such",
    "only", "same", "so", "very", "just", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against",
    "between", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off",
    "over", "under", "again", "further", "once", "patient", "patients",
    "treatment", "treated", "condition", "symptoms", "diagnosis",
}
