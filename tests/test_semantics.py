"""
Comprehensive Test Suite for NexusAI Computational Linguistics & Semantics Module.
Covers:
- WordNetModule (synset retrieval, POS mapping, Lesk WSD, similarity WSD, content extraction)
- LLMSenseSelector (Ollama structured JSON mapping, schema validation, fallback)
- SemanticParser (ERG MRS generation, DELPH-IN formatting, FOL conversions)
- SemanticsService (Singleton, façade methods, RAG query expansion)
- Agent & Pipeline Integration (SemanticsAgent, Orchestrator routing, RAGAgent)
"""

import unittest
import sys
import os
import json

# Ensure workspace root is in PYTHONPATH
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from Backend.Core.Semantics.WordNetModule import WordNetModule
from Backend.Core.Semantics.LLMSenseSelector import LLMSenseSelector
from Backend.Core.Semantics.SemanticParser import SemanticParser, MRS, ElementaryPredication, HandleConstraint
from Backend.Core.Semantics.SemanticsService import SemanticsService
from Backend.Core.Agents.SemanticsAgent import SemanticsAgent
from Backend.Core.Agents.Orchestrator import Orchestrator
from Backend.Core.Agents.RAGAgent import RAGAgent
from Backend.Core.EventBus import event_bus


class TestWordNetModule(unittest.TestCase):
    """Unit tests for WordNetModule."""

    def setUp(self):
        self.wn = WordNetModule()

    def test_synset_retrieval_noun(self):
        synsets = self.wn.get_synsets("bank", pos="n")
        self.assertIsInstance(synsets, list)
        self.assertGreater(len(synsets), 0)
        first = synsets[0]
        self.assertIn("synset_id", first)
        self.assertIn("definition", first)
        self.assertIn("lemmas", first)
        self.assertEqual(first["pos"], "n")

    def test_synset_retrieval_verb(self):
        synsets = self.wn.get_synsets("run", pos="v")
        self.assertGreater(len(synsets), 0)
        self.assertEqual(synsets[0]["pos"], "v")

    def test_unknown_word_handling(self):
        synsets = self.wn.get_synsets("xyznonexistentword98765")
        self.assertEqual(synsets, [])

    def test_get_synset_by_id(self):
        syn = self.wn.get_synset_by_id("bank.n.01")
        self.assertIsNotNone(syn)
        self.assertEqual(syn["synset_id"], "bank.n.01")
        self.assertIn("sloping land", syn["definition"].lower())

    def test_lesk_wsd_river_bank(self):
        sentence = "The river bank overflowed after the heavy rainfall."
        res = self.wn.disambiguate_lesk(sentence, "bank")
        self.assertEqual(res["word"], "bank")
        self.assertIsNotNone(res["best_synset"])
        self.assertGreaterEqual(res["score"], 1)
        # Should match the geological bank sense (bank.n.01)
        self.assertEqual(res["best_synset"], "bank.n.01")
        self.assertTrue(any(term in res["overlap_words"] for term in ["river", "overflowed", "rainfall", "heavy"]))

    def test_lesk_wsd_financial_bank(self):
        sentence = "The commercial bank approved the mortgage loan and deposit."
        res = self.wn.disambiguate_lesk(sentence, "bank")
        self.assertEqual(res["word"], "bank")
        self.assertIsNotNone(res["best_synset"])
        self.assertGreaterEqual(res["score"], 1)
        # Should match financial institution sense
        self.assertIn("financial", res["definition"].lower())

    def test_similarity_wsd(self):
        sentence = "The dog barked at the black cat in the yard."
        res = self.wn.disambiguate_similarity(sentence, "dog")
        self.assertIsNotNone(res["best_synset"])
        self.assertIn("similarity_score", res)
        self.assertGreaterEqual(res["similarity_score"], 0.0)

    def test_content_words_extraction(self):
        sentence = "The fierce tiger chased a fast rabbit through the dense jungle."
        content_words = self.wn.extract_content_words(sentence)
        self.assertIsInstance(content_words, list)
        words_only = [w for w, pos in content_words]
        self.assertIn("tiger", words_only)
        self.assertIn("chased", words_only)
        self.assertIn("rabbit", words_only)


class TestLLMSenseSelector(unittest.TestCase):
    """Unit tests for LLMSenseSelector (Structured JSON Sense Selection)."""

    def setUp(self):
        self.selector = LLMSenseSelector()

    def test_structured_json_schema(self):
        sentence = "The financial bank approved the commercial mortgage loan."
        result = self.selector.select_senses(sentence, target_words=["bank"])
        
        self.assertIsInstance(result, dict)
        self.assertIn("sentence", result)
        self.assertIn("model_used", result)
        self.assertIn("disambiguations", result)
        self.assertEqual(len(result["disambiguations"]), 1)

        d = result["disambiguations"][0]
        required_keys = ["word", "synset_id", "sense_number", "pos", "definition", "confidence", "justification", "method"]
        for key in required_keys:
            self.assertIn(key, d, f"Missing key '{key}' in disambiguation result")

        self.assertEqual(d["word"], "bank")
        self.assertIn("financial", d["definition"].lower())
        self.assertIsInstance(d["confidence"], float)
        self.assertGreater(d["confidence"], 0.0)

    def test_json_extractor_resilience(self):
        # Case 1: Markdown code fences
        text_with_fences = '```json\n{"disambiguations": [{"word": "test", "synset_id": "test.n.01"}]}\n```'
        parsed1 = self.selector._extract_json(text_with_fences)
        self.assertIsNotNone(parsed1)
        self.assertIn("disambiguations", parsed1)

        # Case 2: Plain JSON surrounded by conversational text
        text_with_chatter = 'Here is the requested output:\n{"disambiguations": []}\nHope that helps!'
        parsed2 = self.selector._extract_json(text_with_chatter)
        self.assertIsNotNone(parsed2)
        self.assertIn("disambiguations", parsed2)

        # Case 3: Invalid text
        invalid_text = "This is not json at all."
        parsed3 = self.selector._extract_json(invalid_text)
        self.assertIsNone(parsed3)


class TestSemanticParser(unittest.TestCase):
    """Unit tests for SemanticParser (ERG Minimal Recursion Semantics & FOL)."""

    def setUp(self):
        self.parser = SemanticParser()

    def test_mrs_intransitive_universal(self):
        sentence = "Every dog barks"
        mrs = self.parser.parse(sentence)

        self.assertIsInstance(mrs, MRS)
        self.assertEqual(mrs.ltop, "h0")
        self.assertEqual(mrs.index, "e2")

        pred_names = [ep.pred for ep in mrs.rels]
        self.assertIn("_every_q_rel", pred_names)
        self.assertIn("_dog_n_1_rel", pred_names)
        self.assertIn("_bark_v_1_rel", pred_names)

        # Check HCONS
        hcons_strs = [hc.to_delphin_str() for hc in mrs.hcons]
        self.assertTrue(any("h0 =q h1" in s for s in hcons_strs))

        # Check DELPH-IN format output
        delphin_str = mrs.to_delphin_str()
        self.assertIn("[ LTOP: h0", delphin_str)
        self.assertIn("RELS: <", delphin_str)
        self.assertIn("HCONS: <", delphin_str)

    def test_mrs_transitive_with_adjectives(self):
        sentence = "The black cat chased a mouse"
        mrs = self.parser.parse(sentence)

        pred_names = [ep.pred for ep in mrs.rels]
        self.assertIn("_the_q_rel", pred_names)
        self.assertIn("_black_a_1_rel", pred_names)
        self.assertIn("_cat_n_1_rel", pred_names)
        self.assertIn("_a_q_rel", pred_names)
        self.assertIn("_mouse_n_1_rel", pred_names)
        self.assertIn("_chase_v_1_rel", pred_names)

        # Verify verb arguments
        verb_ep = next(ep for ep in mrs.rels if ep.pred == "_chase_v_1_rel")
        self.assertIn("ARG1", verb_ep.args)
        self.assertIn("ARG2", verb_ep.args)

    def test_fol_conversion_universal(self):
        sentence = "Every dog barks"
        mrs = self.parser.parse(sentence)
        fol = self.parser.convert_to_fol(mrs)

        self.assertIn("fol_formula", fol)
        self.assertIn("fol_ascii", fol)
        self.assertIn("resolution_steps", fol)

        # Universal formula should contain ∀ and → (or all and ->)
        self.assertIn("∀", fol["fol_formula"])
        self.assertIn("→", fol["fol_formula"])
        self.assertIn("all", fol["fol_ascii"])
        self.assertIn("->", fol["fol_ascii"])
        self.assertIn("dog", fol["fol_formula"])
        self.assertIn("bark", fol["fol_formula"])

    def test_fol_conversion_existential(self):
        sentence = "The black cat chased a mouse"
        mrs = self.parser.parse(sentence)
        fol = self.parser.convert_to_fol(mrs)

        self.assertIn("∃", fol["fol_formula"])
        self.assertIn("∧", fol["fol_formula"])
        self.assertIn("exists", fol["fol_ascii"])
        self.assertIn("&", fol["fol_ascii"])
        self.assertIn("cat", fol["fol_formula"])
        self.assertIn("mouse", fol["fol_formula"])
        self.assertIn("chase", fol["fol_formula"])

    def test_fol_conversion_negative(self):
        sentence = "No bird swims"
        mrs = self.parser.parse(sentence)
        fol = self.parser.convert_to_fol(mrs)

        self.assertTrue("¬∃" in fol["fol_formula"] or "~exists" in fol["fol_ascii"])
        self.assertIn("bird", fol["fol_formula"])
        self.assertIn("swim", fol["fol_formula"])


class TestSemanticsService(unittest.TestCase):
    """Unit tests for SemanticsService façade & RAG enrichment."""

    def setUp(self):
        self.service = SemanticsService()

    def test_singleton_integrity(self):
        s1 = SemanticsService()
        s2 = SemanticsService()
        self.assertIs(s1, s2)

    def test_full_semantic_analysis(self):
        sentence = "Every dog barks"
        analysis = self.service.analyze_full_semantics(sentence, use_llm=False)

        self.assertIn("sentence", analysis)
        self.assertIn("wsd", analysis)
        self.assertIn("mrs", analysis)
        self.assertIn("mrs_delphin", analysis)
        self.assertIn("first_order_logic", analysis)

    def test_rag_query_enrichment(self):
        query = "financial bank compliance notes"
        enriched = self.service.enrich_rag_query(query)

        self.assertIn("original_query", enriched)
        self.assertIn("expanded_query", enriched)
        self.assertIn("expansion_terms", enriched)
        self.assertIn("disambiguated_synsets", enriched)

        self.assertEqual(enriched["original_query"], query)
        self.assertGreater(len(enriched["expansion_terms"]), 0)
        self.assertGreater(len(enriched["disambiguated_synsets"]), 0)


class TestAgentAndPipelineIntegration(unittest.TestCase):
    """Integration tests verifying multi-agent routing and RAG pipelines."""

    def setUp(self):
        self.orchestrator = Orchestrator()
        self.semantics_agent = SemanticsAgent()

    def test_semantics_agent_synsets_query(self):
        res = self.semantics_agent.run("synsets for computer")
        self.assertIn("### 📚 WordNet Synsets", res)
        self.assertIn("computer.n.01", res)

    def test_semantics_agent_wsd_query(self):
        res = self.semantics_agent.run("disambiguate the word bank in: The river bank overflowed")
        self.assertIn("### 🎯 Word Sense Disambiguation", res)
        self.assertIn("bank.n.01", res)

    def test_semantics_agent_mrs_query(self):
        res = self.semantics_agent.run("parse to mrs: Every dog barks")
        self.assertIn("### 🧬 ERG-Style Minimal Recursion Semantics", res)
        self.assertIn("_every_q_rel", res)
        self.assertIn("h0 =q h1", res)

    def test_semantics_agent_fol_query(self):
        res = self.semantics_agent.run("convert to first-order logic: The cat chased a mouse")
        self.assertIn("### 📐 First-Order Logic", res)
        self.assertIn("Formal Formula", res)

    def test_orchestrator_routing_to_semantics_agent(self):
        captured_messages = []
        event_bus.on_message.connect(lambda sender, msg: captured_messages.append((sender, msg)))

        self.orchestrator.process("parse to mrs: Every dog barks")
        self.assertGreater(len(captured_messages), 0)
        sender, msg = captured_messages[-1]
        self.assertEqual(sender, "Assistant")
        self.assertIn("Minimal Recursion Semantics", msg)

    def test_rag_agent_semantic_expansion_pipeline(self):
        rag = RAGAgent()
        self.assertTrue(hasattr(rag, "semantics"))
        self.assertIsInstance(rag.semantics, SemanticsService)


if __name__ == "__main__":
    unittest.main()
