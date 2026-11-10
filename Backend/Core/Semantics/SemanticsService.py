"""
Unified Semantics and Computational Linguistics Service for NexusAI.
Coordinates WordNet WSD, LLM-driven sense selection, ERG MRS semantic parsing,
and ChromaDB/RAG query enrichment.
"""

from typing import List, Dict, Any, Optional
from .WordNetModule import WordNetModule
from .LLMSenseSelector import LLMSenseSelector
from .SemanticParser import SemanticParser, MRS


class SemanticsService:
    """
    Unified façade service exposing computational linguistics and semantic processing
    to the rest of NexusAI (Agents, RAG, GUI, and CLI).
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SemanticsService, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = "llama3"):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self.wordnet = WordNetModule()
        self.llm_selector = LLMSenseSelector(model_name=model_name, wordnet=self.wordnet)
        self.parser = SemanticParser()
        self._initialized = True

    # 1. WordNet Capabilities
    def get_synsets(self, word: str, pos: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves synset definitions and relations for a word."""
        return self.wordnet.get_synsets(word, pos=pos)

    def disambiguate_lesk(self, sentence: str, target_word: str, pos: Optional[str] = None) -> Dict[str, Any]:
        """Performs algorithmic Word Sense Disambiguation using Lesk algorithm."""
        return self.wordnet.disambiguate_lesk(sentence, target_word, pos=pos)

    # 2. LLM Sense Selection
    def select_senses_llm(
        self,
        sentence: str,
        target_words: Optional[List[str]] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Disambiguates words in context using local Ollama inference, returning structured JSON.
        """
        return self.llm_selector.select_senses(sentence, target_words=target_words, model=model)

    # 3. ERG MRS and First-Order Logic
    def parse_mrs(self, sentence: str) -> MRS:
        """Parses an English sentence into an ERG-style MRS object."""
        return self.parser.parse(sentence)

    def convert_to_fol(self, sentence_or_mrs) -> Dict[str, Any]:
        """Converts a sentence or an MRS object into a First-Order Logic formula."""
        if isinstance(sentence_or_mrs, str):
            mrs = self.parser.parse(sentence_or_mrs)
        else:
            mrs = sentence_or_mrs
        return self.parser.convert_to_fol(mrs)

    # 4. Full Semantic Pipeline Analysis
    def analyze_full_semantics(
        self,
        sentence: str,
        target_words: Optional[List[str]] = None,
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """
        Runs comprehensive linguistic and semantic analysis:
        - WordNet synset retrieval
        - WSD (LLM-driven or algorithmic)
        - ERG-style MRS representation
        - First-Order Logic (FOL) conversion
        """
        # Step A: Disambiguation
        if use_llm:
            wsd_result = self.select_senses_llm(sentence, target_words=target_words)
        else:
            # Algorithmic WSD for each target or content word
            words = target_words or [w for w, _ in self.wordnet.extract_content_words(sentence)]
            wsd_disambiguations = []
            for w in words:
                lesk = self.wordnet.disambiguate_lesk(sentence, w)
                wsd_disambiguations.append({
                    "word": w,
                    "synset_id": lesk["best_synset"],
                    "definition": lesk["definition"],
                    "score": lesk["score"],
                    "method": "lesk"
                })
            wsd_result = {
                "sentence": sentence,
                "disambiguations": wsd_disambiguations
            }

        # Step B: Semantic Parsing (MRS)
        mrs = self.parser.parse(sentence)

        # Step C: FOL Conversion
        fol = self.parser.convert_to_fol(mrs)

        return {
            "sentence": sentence,
            "wsd": wsd_result,
            "mrs": mrs.to_dict(),
            "mrs_delphin": mrs.to_delphin_str(),
            "first_order_logic": fol
        }

    # 5. ChromaDB / RAG Semantic Integration
    def enrich_rag_query(self, query: str) -> Dict[str, Any]:
        """
        Enriches a user query with disambiguated WordNet synsets and synonyms
        to enhance semantic recall during vector database (ChromaDB) retrieval.

        Args:
            query: Raw user query (e.g. 'notes about river bank risks')

        Returns:
            Dict containing expanded query, extracted synsets, and expansion terms.
        """
        content_words = self.wordnet.extract_content_words(query)
        expansion_terms = set()
        synsets_found = []

        for word, pos in content_words:
            # Use Lesk to find the most contextually relevant sense in the query
            lesk_res = self.wordnet.disambiguate_lesk(query, word, pos=pos)
            best_id = lesk_res.get("best_synset")
            if best_id:
                syn_info = self.wordnet.get_synset_by_id(best_id)
                if syn_info:
                    synsets_found.append({
                        "word": word,
                        "synset_id": best_id,
                        "definition": syn_info["definition"]
                    })
                    # Add synonyms (lemmas) as expansion candidates
                    for lemma in syn_info.get("lemmas", []):
                        if lemma.lower() != word.lower():
                            expansion_terms.add(lemma.lower())

        expanded_query = query
        if expansion_terms:
            expanded_query = f"{query} ({', '.join(sorted(expansion_terms))})"

        return {
            "original_query": query,
            "expanded_query": expanded_query,
            "expansion_terms": list(expansion_terms),
            "disambiguated_synsets": synsets_found
        }
