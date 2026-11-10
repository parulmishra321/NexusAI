"""
LLM-Driven Word Sense Selection Module for NexusAI.
Uses local Ollama inference to perform Word Sense Disambiguation (WSD)
and returns structured JSON mapping words to exact WordNet synset IDs and definitions.
"""

import json
import re
from typing import List, Dict, Any, Optional
from .WordNetModule import WordNetModule

try:
    import ollama
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False
    ollama = None


class LLMSenseSelector:
    """
    Selects WordNet synsets for words in sentence context using local Ollama LLMs,
    with robust structured JSON parsing and algorithmic fallback.
    """

    def __init__(self, model_name: str = "llama3", wordnet: Optional[WordNetModule] = None):
        self.model_name = model_name
        self.wordnet = wordnet or WordNetModule()

    def select_senses(
        self,
        sentence: str,
        target_words: Optional[List[str]] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Disambiguates words in context using Ollama inference, returning structured JSON.

        Args:
            sentence: The context sentence.
            target_words: Specific words to disambiguate. If None, content words are extracted.
            model: Optional model override.

        Returns:
            Structured JSON dict conforming to:
            {
                "sentence": str,
                "model_used": str,
                "disambiguations": [
                    {
                        "word": str,
                        "synset_id": str,
                        "sense_number": int,
                        "pos": str,
                        "definition": str,
                        "confidence": float,
                        "justification": str
                    }
                ]
            }
        """
        active_model = model or self.model_name

        # 1. Identify words to disambiguate
        if not target_words:
            content_words = self.wordnet.extract_content_words(sentence)
            # Filter words to those with multiple synsets or significant ambiguity
            target_words = []
            for word, pos in content_words:
                synsets = self.wordnet.get_synsets(word, pos=pos)
                if len(synsets) > 0:
                    target_words.append(word)

        if not target_words:
            return {
                "sentence": sentence,
                "model_used": active_model,
                "disambiguations": []
            }

        # 2. Collect candidate synsets for each target word
        candidates_by_word: Dict[str, List[Dict[str, Any]]] = {}
        for word in target_words:
            syns = self.wordnet.get_synsets(word)
            if syns:
                # Limit to top 6 senses per word to keep prompt concise
                candidates_by_word[word] = syns[:6]

        if not candidates_by_word:
            return {
                "sentence": sentence,
                "model_used": active_model,
                "disambiguations": []
            }

        # 3. Try LLM sense selection via Ollama
        if _OLLAMA_AVAILABLE:
            try:
                llm_result = self._query_ollama(sentence, candidates_by_word, active_model)
                if llm_result and "disambiguations" in llm_result:
                    return self._validate_and_enrich_result(llm_result, sentence, candidates_by_word, active_model)
            except Exception as e:
                # Log or note error, fall back to algorithmic WSD
                pass

        # 4. Fallback to algorithmic Lesk WSD
        return self._algorithmic_fallback(sentence, candidates_by_word, active_model)

    def _query_ollama(
        self,
        sentence: str,
        candidates_by_word: Dict[str, List[Dict[str, Any]]],
        model: str
    ) -> Optional[Dict[str, Any]]:
        """Prompts Ollama to select the best WordNet synset from candidates."""
        
        candidates_text = ""
        for word, synsets in candidates_by_word.items():
            candidates_text += f"\nWord: '{word}'\nCandidate WordNet Senses:\n"
            for i, s in enumerate(synsets, 1):
                examples_str = f" (e.g., '{s['examples'][0]}')" if s['examples'] else ""
                candidates_text += f"  [{i}] ID: {s['synset_id']} | POS: {s['pos']} | Def: {s['definition']}{examples_str}\n"

        system_prompt = (
            "You are an expert computational linguist specializing in lexical semantics and Word Sense Disambiguation (WSD).\n"
            "Your task is to analyze the target sentence and assign the exact correct WordNet synset ID to each requested word.\n"
            "You MUST select only from the provided Candidate WordNet Senses.\n"
            "Return STRICT JSON only, matching this exact structure without markdown commentary:\n"
            "{\n"
            '  "disambiguations": [\n'
            "    {\n"
            '      "word": "<word>",\n'
            '      "synset_id": "<exact synset_id from candidates>",\n'
            '      "sense_number": <integer index from candidates>,\n'
            '      "pos": "<noun/verb/adj/adv>",\n'
            '      "definition": "<definition from candidate>",\n'
            '      "confidence": <float between 0.0 and 1.0>,\n'
            '      "justification": "<concise semantic explanation based on sentence context>"\n'
            "    }\n"
            "  ]\n"
            "}"
        )

        user_prompt = (
            f"Sentence: \"{sentence}\"\n\n"
            f"Candidate senses to choose from:\n{candidates_text}\n"
            "Output the strict JSON mapping for these words:"
        )

        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": 0.1}
        )

        raw_content = response['message']['content'].strip()
        return self._extract_json(raw_content)

    def _extract_json(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """Extracts JSON object from text, handling markdown fences and prefixes."""
        # Remove markdown code blocks if present
        cleaned = re.sub(r'^```(?:json)?\s*', '', raw_text, flags=re.MULTILINE)
        cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE).strip()

        # Direct parse attempt
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Search for first JSON block with regex
        match = re.search(r'(\{[\s\S]*\})', raw_text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        return None

    def _validate_and_enrich_result(
        self,
        llm_data: Dict[str, Any],
        sentence: str,
        candidates_by_word: Dict[str, List[Dict[str, Any]]],
        model: str
    ) -> Dict[str, Any]:
        """Validates that synset IDs exist in candidate list and enriches missing fields."""
        enriched = []
        raw_disambiguations = llm_data.get("disambiguations", [])

        # Map by word
        output_by_word = {d.get("word", "").lower(): d for d in raw_disambiguations if isinstance(d, dict)}

        for word, cand_list in candidates_by_word.items():
            valid_ids = {c["synset_id"]: c for c in cand_list}
            item = output_by_word.get(word.lower())

            if item and item.get("synset_id") in valid_ids:
                matched_cand = valid_ids[item["synset_id"]]
                enriched.append({
                    "word": word,
                    "synset_id": matched_cand["synset_id"],
                    "sense_number": item.get("sense_number", 1),
                    "pos": matched_cand["pos"],
                    "definition": matched_cand["definition"],
                    "confidence": round(float(item.get("confidence", 0.95)), 2),
                    "justification": item.get("justification", "Contextually matched based on sentence context."),
                    "method": "llm_ollama"
                })
            else:
                # Fallback to top candidate or Lesk for this specific word
                lesk_res = self.wordnet.disambiguate_lesk(sentence, word)
                best_id = lesk_res["best_synset"] or cand_list[0]["synset_id"]
                cand_meta = self.wordnet.get_synset_by_id(best_id) or cand_list[0]
                enriched.append({
                    "word": word,
                    "synset_id": cand_meta["synset_id"],
                    "sense_number": 1,
                    "pos": cand_meta["pos"],
                    "definition": cand_meta["definition"],
                    "confidence": 0.80,
                    "justification": f"Algorithmic overlap score: {lesk_res['score']}",
                    "method": "algorithmic_fallback"
                })

        return {
            "sentence": sentence,
            "model_used": model,
            "disambiguations": enriched
        }

    def _algorithmic_fallback(
        self,
        sentence: str,
        candidates_by_word: Dict[str, List[Dict[str, Any]]],
        model: str
    ) -> Dict[str, Any]:
        """Performs structured sense selection via Lesk algorithm when LLM is unavailable."""
        disambiguations = []
        for word, cand_list in candidates_by_word.items():
            lesk_res = self.wordnet.disambiguate_lesk(sentence, word)
            best_id = lesk_res["best_synset"] or cand_list[0]["synset_id"]
            cand_meta = self.wordnet.get_synset_by_id(best_id) or cand_list[0]

            disambiguations.append({
                "word": word,
                "synset_id": cand_meta["synset_id"],
                "sense_number": 1,
                "pos": cand_meta["pos"],
                "definition": cand_meta["definition"],
                "confidence": 0.85 if lesk_res["score"] > 0 else 0.70,
                "justification": (
                    f"Selected via Lesk WSD with overlap tokens: {lesk_res['overlap_words']}"
                    if lesk_res["overlap_words"] else "Default primary WordNet sense selected."
                ),
                "method": "algorithmic_lesk_fallback"
            })

        return {
            "sentence": sentence,
            "model_used": f"{model} (offline/algorithmic fallback)",
            "disambiguations": disambiguations
        }
