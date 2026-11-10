"""
Semantics Agent for NexusAI.
Processes natural language queries requiring computational linguistics,
WordNet Word Sense Disambiguation (WSD), ERG Minimal Recursion Semantics (MRS),
and First-Order Logic (FOL) conversion.
"""

from .BaseAgent import BaseAgent
from ..EventBus import event_bus
from ..Semantics.SemanticsService import SemanticsService
import json
import re


class SemanticsAgent(BaseAgent):
    """
    Agent dedicated to computational semantics, WordNet lookups,
    WSD, ERG-style MRS semantic parsing, and First-Order Logic conversion.
    """

    def __init__(self):
        super().__init__("SemanticsAgent")
        self.service = SemanticsService()

    def run(self, query: str) -> str:
        """
        Executes semantic processing based on user input.
        """
        query_lower = query.lower()
        event_bus.on_status_change.emit("Processing Semantics...")

        try:
            # 1. MRS Parsing Request
            if "mrs" in query_lower or "minimal recursion" in query_lower:
                sentence = self._extract_sentence(query)
                event_bus.on_status_change.emit("Parsing ERG MRS...")
                mrs = self.service.parse_mrs(sentence)
                fol = self.service.convert_to_fol(mrs)
                
                return (
                    f"### 🧬 ERG-Style Minimal Recursion Semantics (MRS)\n\n"
                    f"**Sentence:** \"{sentence}\"\n\n"
                    f"```text\n{mrs.to_delphin_str()}\n```\n\n"
                    f"**First-Order Logic (FOL):**\n"
                    f"- **Formula:** `{fol['fol_formula']}`\n"
                    f"- **ASCII:** `{fol['fol_ascii']}`"
                )

            # 2. First-Order Logic (FOL) Conversion Request
            if "first-order logic" in query_lower or "first order logic" in query_lower or "fol" in query_lower:
                sentence = self._extract_sentence(query)
                event_bus.on_status_change.emit("Synthesizing First-Order Logic...")
                fol = self.service.convert_to_fol(sentence)
                mrs = self.service.parse_mrs(sentence)

                steps_md = "\n".join([f"- {s}" for s in fol["resolution_steps"]])
                return (
                    f"### 📐 First-Order Logic (FOL) Representation\n\n"
                    f"**Sentence:** \"{sentence}\"\n\n"
                    f"**Formal Formula:**\n"
                    f"```\n{fol['fol_formula']}\n```\n"
                    f"**ASCII Notation:** `{fol['fol_ascii']}`\n\n"
                    f"**Scoping & Resolution Steps:**\n{steps_md}\n\n"
                    f"**Underlying MRS Index:** `{fol['mrs_index']}`"
                )

            # 3. WordNet Synset Retrieval Request
            if "synset" in query_lower or "lookup" in query_lower or "wordnet" in query_lower:
                target_word = self._extract_target_word(query)
                event_bus.on_status_change.emit(f"Retrieving Synsets for '{target_word}'...")
                synsets = self.service.get_synsets(target_word)

                if not synsets:
                    return f"No WordNet synsets found for word '{target_word}'."

                md_out = f"### 📚 WordNet Synsets for '{target_word}'\n\n"
                md_out += f"Found **{len(synsets)}** synsets:\n\n"
                for i, s in enumerate(synsets[:6], 1):
                    lemmas_str = ", ".join(s['lemmas'])
                    ex_str = f" *(\"{s['examples'][0]}\")*" if s['examples'] else ""
                    md_out += f"{i}. **`{s['synset_id']}`** ({s['pos']}): {s['definition']}{ex_str}\n"
                    md_out += f"   - *Synonyms:* {lemmas_str}\n"
                return md_out

            # 4. Word Sense Disambiguation (WSD) / Sense Selection
            if "disambiguate" in query_lower or "word sense" in query_lower or "wsd" in query_lower or "sense" in query_lower:
                sentence = self._extract_sentence(query)
                target_word = self._extract_target_word(query, default=None)
                target_words = [target_word] if target_word else None

                event_bus.on_status_change.emit("Selecting Word Senses...")
                result = self.service.select_senses_llm(sentence, target_words=target_words)

                disambiguations = result.get("disambiguations", [])
                if not disambiguations:
                    return f"Could not disambiguate target words in: \"{sentence}\""

                md_out = f"### 🎯 Word Sense Disambiguation (WSD)\n\n"
                md_out += f"**Sentence:** \"{sentence}\"\n"
                md_out += f"**Engine:** `{result.get('model_used', 'Ollama / WordNet')}`\n\n"
                md_out += "| Target Word | Selected Synset ID | Definition | Confidence | Justification |\n"
                md_out += "| :--- | :--- | :--- | :--- | :--- |\n"
                for d in disambiguations:
                    md_out += f"| **{d['word']}** | `{d['synset_id']}` | {d['definition']} | {d.get('confidence', 0.9):.2f} | {d.get('justification', 'N/A')} |\n"

                md_out += "\n**Structured JSON Output:**\n"
                md_out += f"```json\n{json.dumps(result, indent=2)}\n```"
                return md_out

            # 5. Default / Comprehensive Analysis
            sentence = self._extract_sentence(query)
            event_bus.on_status_change.emit("Running Full Semantic Pipeline...")
            analysis = self.service.analyze_full_semantics(sentence, use_llm=True)

            fol = analysis["first_order_logic"]
            wsd_items = analysis["wsd"].get("disambiguations", [])

            md_out = f"### 🔬 Comprehensive Semantic Analysis\n\n"
            md_out += f"**Input Sentence:** \"{sentence}\"\n\n"
            md_out += f"#### 1. First-Order Logic (FOL)\n"
            md_out += f"- **Formula:** `{fol['fol_formula']}`\n"
            md_out += f"- **ASCII:** `{fol['fol_ascii']}`\n\n"
            
            md_out += f"#### 2. Word Sense Disambiguation (WSD)\n"
            for d in wsd_items:
                md_out += f"- **{d['word']}** -> `{d['synset_id']}`: {d['definition']} *(Conf: {d.get('confidence', 0.85)})*\n"

            md_out += f"\n#### 3. ERG-Style Minimal Recursion Semantics (MRS)\n"
            md_out += f"```text\n{analysis['mrs_delphin']}\n```\n"

            return md_out

        except Exception as e:
            return f"Semantics Processing Error: {str(e)}"

    def _extract_sentence(self, query: str) -> str:
        """Extracts the sentence wrapped in quotes or following a colon/marker."""
        # Check for quoted text
        match = re.search(r'["\']([^"\']+)["\']', query)
        if match:
            return match.group(1).strip()

        # Check for colon
        if ":" in query:
            parts = query.split(":", 1)
            candidate = parts[1].strip()
            if len(candidate) > 3:
                return candidate

        # Clean command prefixes
        prefixes = [
            "parse to mrs", "parse mrs", "mrs for", "convert to first order logic",
            "convert to fol", "fol for", "disambiguate", "word sense for",
            "analyze semantics for", "analyze semantics"
        ]
        cleaned = query
        for p in prefixes:
            if cleaned.lower().startswith(p):
                cleaned = cleaned[len(p):].strip()
                break

        return cleaned if cleaned else query

    def _extract_target_word(self, query: str, default: str = "bank") -> str:
        """Extracts target word from user query."""
        match = re.search(r"(?:word|synset|for|disambiguate)\s+['\"]?([a-zA-Z_-]+)['\"]?", query, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            if candidate.lower() not in ('to', 'for', 'the', 'a', 'this', 'that', 'word', 'mrs', 'fol'):
                return candidate
        return default
