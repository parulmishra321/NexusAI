"""
WordNet Integration Module for NexusAI.
Provides Word Sense Disambiguation (WSD), synset retrieval, and semantic relations
using NLTK's WordNet corpus.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import re

try:
    import nltk
    from nltk.corpus import wordnet as wn
    from nltk.corpus import stopwords
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False
    wn = None
    stopwords = None


class WordNetModule:
    """
    Interface for WordNet synset retrieval, lexical relations,
    and algorithmic Word Sense Disambiguation (Lesk & Semantic Similarity).
    """

    POS_MAP = {
        'n': wn.NOUN if _NLTK_AVAILABLE else 'n',
        'v': wn.VERB if _NLTK_AVAILABLE else 'v',
        'a': wn.ADJ if _NLTK_AVAILABLE else 'a',
        's': wn.ADJ_SAT if _NLTK_AVAILABLE else 's',
        'r': wn.ADV if _NLTK_AVAILABLE else 'r',
        # Penn Treebank tag prefix mapping
        'NN': wn.NOUN if _NLTK_AVAILABLE else 'n',
        'VB': wn.VERB if _NLTK_AVAILABLE else 'v',
        'JJ': wn.ADJ if _NLTK_AVAILABLE else 'a',
        'RB': wn.ADV if _NLTK_AVAILABLE else 'r'
    }

    def __init__(self):
        self._ensure_resources()
        self._stopwords_set = self._load_stopwords()

    def _ensure_resources(self):
        """Ensures that WordNet and supporting corpora are loaded."""
        if not _NLTK_AVAILABLE:
            return
        try:
            wn.synsets('entity')
        except (LookupError, AttributeError):
            try:
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
            except Exception:
                pass

    def _load_stopwords(self) -> Set[str]:
        """Loads English stopwords set with fallback."""
        default_stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
            "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
            'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
            'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
            'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
            'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
            'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
            'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', "now"
        }
        if _NLTK_AVAILABLE and stopwords:
            try:
                return set(stopwords.words('english'))
            except Exception:
                return default_stopwords
        return default_stopwords

    def map_pos(self, pos: Optional[str]) -> Optional[str]:
        """Maps an arbitrary POS tag to WordNet POS constants."""
        if not pos:
            return None
        pos = pos.lower()
        if pos in ('n', 'noun'):
            return wn.NOUN if _NLTK_AVAILABLE else 'n'
        if pos in ('v', 'verb'):
            return wn.VERB if _NLTK_AVAILABLE else 'v'
        if pos in ('a', 'adj', 'adjective'):
            return wn.ADJ if _NLTK_AVAILABLE else 'a'
        if pos in ('r', 'adv', 'adverb'):
            return wn.ADV if _NLTK_AVAILABLE else 'r'
        
        # Check prefix in POS_MAP
        for prefix, mapped in self.POS_MAP.items():
            if pos.upper().startswith(prefix):
                return mapped
        return None

    def get_synsets(self, word: str, pos: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves all WordNet synsets for a given word/lemma.
        
        Args:
            word: Target word or compound (e.g. 'bank', 'hot dog').
            pos: Optional part of speech filter ('n', 'v', 'a', 'r').

        Returns:
            List of structured synset dictionaries.
        """
        if not _NLTK_AVAILABLE or wn is None:
            return []

        clean_word = word.strip().lower().replace(" ", "_")
        wn_pos = self.map_pos(pos)

        try:
            raw_synsets = wn.synsets(clean_word, pos=wn_pos)
            if not raw_synsets:
                # Try finding base form via morphy
                base = wn.morphy(clean_word, pos=wn_pos) if wn_pos else wn.morphy(clean_word)
                if base and base != clean_word:
                    raw_synsets = wn.synsets(base, pos=wn_pos)
        except Exception:
            raw_synsets = []

        results = []
        for s in raw_synsets:
            results.append(self._format_synset(s))
        return results

    def get_synset_by_id(self, synset_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single synset by its exact ID (e.g. 'bank.n.01')."""
        if not _NLTK_AVAILABLE or wn is None:
            return None
        try:
            s = wn.synset(synset_id)
            return self._format_synset(s)
        except Exception:
            return None

    def _format_synset(self, s) -> Dict[str, Any]:
        """Formats an NLTK Synset into a clean JSON-serializable dictionary."""
        lemmas = [lemma.name().replace('_', ' ') for lemma in s.lemmas()]
        hypernyms = [h.name() for h in s.hypernyms()]
        hyponyms = [h.name() for h in s.hyponyms()[:5]]

        return {
            "synset_id": s.name(),
            "name": s.name(),
            "pos": s.pos(),
            "definition": s.definition(),
            "examples": s.examples(),
            "lemmas": lemmas,
            "hypernyms": hypernyms,
            "hyponyms": hyponyms
        }

    def tokenize_and_filter(self, text: str) -> List[str]:
        """Simple tokenizer that lowercases, removes non-alphanumeric, and filters stopwords."""
        words = re.findall(r'[a-zA-Z0-9]+', text.lower())
        return [w for w in words if w not in self._stopwords_set and len(w) > 1]

    def disambiguate_lesk(
        self,
        sentence: str,
        target_word: str,
        pos: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Simplified Lesk algorithm for Word Sense Disambiguation (WSD).
        Compares overlap between sentence context words and synset definitions + examples.

        Args:
            sentence: Context sentence containing target_word.
            target_word: Word to disambiguate.
            pos: Optional part of speech constraint.

        Returns:
            Dict containing best_synset, definition, score, overlap tokens, and candidate scores.
        """
        candidates = self.get_synsets(target_word, pos=pos)
        if not candidates:
            return {
                "word": target_word,
                "best_synset": None,
                "definition": "No synsets found in WordNet.",
                "score": 0,
                "overlap_words": [],
                "candidates": []
            }

        context_words = set(self.tokenize_and_filter(sentence))
        # Remove the target word itself from context
        context_words.discard(target_word.lower())

        best_candidate = candidates[0]
        max_score = -1
        best_overlap: List[str] = []
        candidate_evaluations = []

        for cand in candidates:
            # Build gloss signature: definition + examples + lemmas
            gloss_text = cand["definition"] + " " + " ".join(cand["examples"]) + " " + " ".join(cand["lemmas"])
            
            # Extended Lesk: include hypernym definitions for additional context
            for hyp_id in cand["hypernyms"]:
                hyp_syn = self.get_synset_by_id(hyp_id)
                if hyp_syn:
                    gloss_text += " " + hyp_syn["definition"]

            gloss_words = set(self.tokenize_and_filter(gloss_text))
            overlap = context_words.intersection(gloss_words)
            score = len(overlap)

            candidate_evaluations.append({
                "synset_id": cand["synset_id"],
                "definition": cand["definition"],
                "score": score,
                "overlap": list(overlap)
            })

            if score > max_score:
                max_score = score
                best_candidate = cand
                best_overlap = list(overlap)

        return {
            "word": target_word,
            "best_synset": best_candidate["synset_id"],
            "definition": best_candidate["definition"],
            "score": max_score,
            "overlap_words": best_overlap,
            "candidates": candidate_evaluations
        }

    def disambiguate_similarity(
        self,
        sentence: str,
        target_word: str,
        pos: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        WSD using WordNet semantic similarity (Wu-Palmer or Path similarity)
        against other content words in the sentence.
        """
        if not _NLTK_AVAILABLE or wn is None:
            return self.disambiguate_lesk(sentence, target_word, pos)

        target_synsets = self.get_synsets(target_word, pos=pos)
        if not target_synsets:
            return self.disambiguate_lesk(sentence, target_word, pos)

        context_tokens = self.tokenize_and_filter(sentence)
        context_tokens = [w for w in context_tokens if w != target_word.lower()]

        # Gather synsets for context words
        context_synsets = []
        for w in context_tokens:
            s_list = wn.synsets(w)
            if s_list:
                context_synsets.append(s_list[0])  # primary sense

        if not context_synsets:
            return self.disambiguate_lesk(sentence, target_word, pos)

        best_score = -1.0
        best_syn = target_synsets[0]

        for cand in target_synsets:
            cand_raw = wn.synset(cand["synset_id"])
            scores = []
            for c_syn in context_synsets:
                try:
                    sim = cand_raw.wup_similarity(c_syn)
                    if sim is not None:
                        scores.append(sim)
                except Exception:
                    pass
            avg_score = sum(scores) / len(scores) if scores else 0.0
            if avg_score > best_score:
                best_score = avg_score
                best_syn = cand

        return {
            "word": target_word,
            "best_synset": best_syn["synset_id"],
            "definition": best_syn["definition"],
            "similarity_score": round(best_score, 4),
            "method": "wu_palmer"
        }

    def extract_content_words(self, sentence: str) -> List[Tuple[str, Optional[str]]]:
        """
        Extracts content words with optional POS tags.
        Returns list of (word, wn_pos).
        """
        tokens = self.tokenize_and_filter(sentence)
        result = []
        
        # Try NLTK POS tagger if available
        tagged = None
        if _NLTK_AVAILABLE:
            try:
                tagged = nltk.pos_tag(tokens)
            except Exception:
                tagged = None

        if tagged:
            for word, tag in tagged:
                wn_pos = self.map_pos(tag)
                # Only keep nouns, verbs, adjectives, and adverbs
                if wn_pos in ('n', 'v', 'a', 'r'):
                    result.append((word, wn_pos))
        else:
            for word in tokens:
                # Fallback: check if word has any synsets in WordNet
                syns = self.get_synsets(word)
                if syns:
                    result.append((word, syns[0]["pos"]))
        return result
