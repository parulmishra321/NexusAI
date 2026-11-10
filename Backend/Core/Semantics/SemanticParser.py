"""
Mock Semantic Parsing Handler for NexusAI.
Implements English Resource Grammar (ERG) style Minimal Recursion Semantics (MRS)
and First-Order Logic (FOL) conversion.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import re

try:
    import nltk
    from nltk.corpus import wordnet as wn
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False
    wn = None


class ElementaryPredication:
    """Represents an ERG-style Elementary Predication (EP) in MRS."""

    def __init__(
        self,
        lbl: str,
        pred: str,
        arg0: str,
        args: Optional[Dict[str, str]] = None,
        cfrom: int = -1,
        cto: int = -1
    ):
        self.lbl = lbl            # Handle label (e.g. 'h1')
        self.pred = pred          # ERG predicate name (e.g. '_dog_n_1_rel')
        self.arg0 = arg0          # Intrinsic variable (e.g. 'x3' or 'e2')
        self.args = args or {}    # Additional args: ARG1, ARG2, RSTR, BODY, CARG, etc.
        self.cfrom = cfrom        # Character start offset
        self.cto = cto            # Character end offset

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "lbl": self.lbl,
            "pred": self.pred,
            "arg0": self.arg0
        }
        res.update(self.args)
        if self.cfrom >= 0:
            res["cfrom"] = self.cfrom
            res["cto"] = self.cto
        return res

    def to_delphin_str(self) -> str:
        """Standard DELPH-IN ASCII representation for an EP."""
        args_str = f"ARG0: {self.arg0}"
        for k, v in self.args.items():
            args_str += f" {k}: {v}"
        return f"[ {self.pred} LBL: {self.lbl} {args_str} ]"

    def __repr__(self):
        return self.to_delphin_str()


class HandleConstraint:
    """Represents an MRS handle constraint (typically qeq: =_q)."""

    def __init__(self, hi: str, relation: str, lo: str):
        self.hi = hi
        self.relation = relation  # e.g. 'qeq'
        self.lo = lo

    def to_dict(self) -> Dict[str, str]:
        return {"hi": self.hi, "relation": self.relation, "lo": self.lo}

    def to_delphin_str(self) -> str:
        return f"{self.hi} =q {self.lo}"

    def __repr__(self):
        return self.to_delphin_str()


class MRS:
    """
    Minimal Recursion Semantics (MRS) data structure conforming to
    English Resource Grammar (ERG) conventions.
    """

    def __init__(
        self,
        ltop: str,
        index: str,
        rels: List[ElementaryPredication],
        hcons: List[HandleConstraint],
        surface_text: str = ""
    ):
        self.ltop = ltop          # Top handle (e.g. 'h0')
        self.index = index        # Main event or entity index (e.g. 'e2')
        self.rels = rels          # List of Elementary Predications
        self.hcons = hcons        # List of Handle Constraints
        self.surface_text = surface_text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "surface_text": self.surface_text,
            "ltop": self.ltop,
            "index": self.index,
            "rels": [ep.to_dict() for ep in self.rels],
            "hcons": [hc.to_dict() for hc in self.hcons]
        }

    def to_delphin_str(self) -> str:
        """Formats the MRS in classic DELPH-IN ASCII format."""
        rels_str = "\n    ".join([ep.to_delphin_str() for ep in self.rels])
        hcons_str = "  ".join([hc.to_delphin_str() for hc in self.hcons])
        return (
            f"[ LTOP: {self.ltop}\n"
            f"  INDEX: {self.index} [ e SF: prop TENSE: pres MOOD: indicative ]\n"
            f"  RELS: <\n    {rels_str}\n  >\n"
            f"  HCONS: < {hcons_str} >\n]"
        )

    def __repr__(self):
        return self.to_delphin_str()


class SemanticParser:
    """
    Mock semantic parser producing ERG-style MRS structures and First-Order Logic (FOL) formulas
    from natural language sentences.
    """

    QUANTIFIER_MAP = {
        'every': ('_every_q_rel', 'universal'),
        'all': ('_all_q_rel', 'universal'),
        'each': ('_each_q_rel', 'universal'),
        'a': ('_a_q_rel', 'existential'),
        'an': ('_a_q_rel', 'existential'),
        'the': ('_the_q_rel', 'existential'),
        'some': ('_some_q_rel', 'existential'),
        'no': ('_no_q_rel', 'negative'),
        'any': ('_any_q_rel', 'universal')
    }

    def __init__(self):
        self._var_counter = 0

    def parse(self, sentence: str) -> MRS:
        """
        Parses an English sentence into an ERG-style MRS representation.

        Args:
            sentence: Input sentence (e.g. "Every dog barks", "The river bank flooded the city").

        Returns:
            MRS object.
        """
        clean_text = sentence.strip().rstrip('.!?')
        words = clean_text.split()
        lower_words = [w.lower() for w in words]

        # Allocate base handles
        ltop = "h0"
        main_event = "e2"
        h_verb = "h1"

        rels: List[ElementaryPredication] = []
        hcons: List[HandleConstraint] = []
        
        # Link top handle to main verb handle (h0 =q h1)
        hcons.append(HandleConstraint(hi=ltop, relation="qeq", lo=h_verb))

        # Perform shallow syntactic analysis (Subject, Verb, Object, Modifiers)
        parsed_components = self._analyze_syntax(words, lower_words)

        handle_idx = 3
        var_idx = 3

        # 1. Process Subject NP
        subj = parsed_components.get("subject", {})
        subj_var = f"x{var_idx}"
        var_idx += 1

        if subj:
            q_word = subj.get("quantifier", "the")
            q_pred, _ = self.QUANTIFIER_MAP.get(q_word, ('_the_q_rel', 'existential'))
            
            q_lbl = f"h{handle_idx}"
            rstr_h = f"h{handle_idx + 1}"
            body_h = f"h{handle_idx + 2}"
            noun_lbl = f"h{handle_idx + 3}"
            handle_idx += 4

            # Quantifier EP
            rels.append(ElementaryPredication(
                lbl=q_lbl,
                pred=q_pred,
                arg0=subj_var,
                args={"RSTR": rstr_h, "BODY": body_h}
            ))

            # Quantifier constraint: RSTR =q noun_lbl
            hcons.append(HandleConstraint(hi=rstr_h, relation="qeq", lo=noun_lbl))

            # Adjectives modifying subject
            for adj in subj.get("adjectives", []):
                adj_pred = f"_{adj}_a_1_rel"
                rels.append(ElementaryPredication(
                    lbl=noun_lbl,
                    pred=adj_pred,
                    arg0=f"e{var_idx}",
                    args={"ARG1": subj_var}
                ))
                var_idx += 1

            # Noun EP
            subj_noun = subj.get("noun", "thing")
            noun_pred = f"_{subj_noun}_n_1_rel" if not subj.get("is_relational") else f"_{subj_noun}_n_of_rel"
            rels.append(ElementaryPredication(
                lbl=noun_lbl,
                pred=noun_pred,
                arg0=subj_var
            ))

        # 2. Process Direct Object NP (if transitive)
        obj = parsed_components.get("object")
        obj_var = None
        if obj:
            obj_var = f"x{var_idx}"
            var_idx += 1

            q_word = obj.get("quantifier", "a")
            q_pred, _ = self.QUANTIFIER_MAP.get(q_word, ('_a_q_rel', 'existential'))
            
            q_lbl = f"h{handle_idx}"
            rstr_h = f"h{handle_idx + 1}"
            body_h = f"h{handle_idx + 2}"
            noun_lbl = f"h{handle_idx + 3}"
            handle_idx += 4

            rels.append(ElementaryPredication(
                lbl=q_lbl,
                pred=q_pred,
                arg0=obj_var,
                args={"RSTR": rstr_h, "BODY": body_h}
            ))
            hcons.append(HandleConstraint(hi=rstr_h, relation="qeq", lo=noun_lbl))

            for adj in obj.get("adjectives", []):
                adj_pred = f"_{adj}_a_1_rel"
                rels.append(ElementaryPredication(
                    lbl=noun_lbl,
                    pred=adj_pred,
                    arg0=f"e{var_idx}",
                    args={"ARG1": obj_var}
                ))
                var_idx += 1

            obj_noun = obj.get("noun", "thing")
            noun_pred = f"_{obj_noun}_n_1_rel"
            rels.append(ElementaryPredication(
                lbl=noun_lbl,
                pred=noun_pred,
                arg0=obj_var
            ))

        # 3. Process Verb EP
        verb_lemma = parsed_components.get("verb", "be")
        verb_pred = f"_{verb_lemma}_v_1_rel"
        verb_args = {"ARG1": subj_var}
        if obj_var:
            verb_args["ARG2"] = obj_var

        rels.append(ElementaryPredication(
            lbl=h_verb,
            pred=verb_pred,
            arg0=main_event,
            args=verb_args
        ))

        # 4. Process Prepositional Phrases (if any)
        for prep_item in parsed_components.get("prepositions", []):
            prep_word = prep_item.get("prep", "in")
            prep_pred = f"_{prep_word}_p_rel"
            prep_target_var = f"x{var_idx}"
            var_idx += 1

            # Noun inside prep phrase
            prep_noun = prep_item.get("noun", "place")
            p_noun_lbl = f"h{handle_idx}"
            handle_idx += 1
            
            rels.append(ElementaryPredication(
                lbl=p_noun_lbl,
                pred=f"_{prep_noun}_n_1_rel",
                arg0=prep_target_var
            ))

            rels.append(ElementaryPredication(
                lbl=h_verb,
                pred=prep_pred,
                arg0=f"e{var_idx}",
                args={"ARG1": main_event, "ARG2": prep_target_var}
            ))
            var_idx += 1

        return MRS(
            ltop=ltop,
            index=main_event,
            rels=rels,
            hcons=hcons,
            surface_text=sentence
        )

    def _analyze_syntax(self, words: List[str], lower_words: List[str]) -> Dict[str, Any]:
        """Heuristic and POS-assisted syntactic chunker."""
        tagged: List[Tuple[str, str]] = []
        if _NLTK_AVAILABLE:
            try:
                tagged = nltk.pos_tag(words)
            except Exception:
                tagged = []

        if not tagged:
            # Simple fallback tagging
            tagged = [(w, 'NN' if w in ('dog', 'bank', 'river', 'city', 'cat', 'mouse') else 'VB') for w in lower_words]

        # Scan for prepositions
        preps = []
        prep_indices = [i for i, (w, _) in enumerate(tagged) if w.lower() in ('in', 'on', 'at', 'with', 'from', 'to', 'after', 'before')]

        # Find main verb index
        verb_idx = -1
        for i, (w, tag) in enumerate(tagged):
            if i in prep_indices:
                continue
            if tag.startswith('VB') or (verb_idx == -1 and i > 0 and w.lower() in ('barks', 'bark', 'flooded', 'chased', 'runs', 'approved', 'is', 'swims', 'answers')):
                verb_idx = i
                break

        if verb_idx == -1:
            verb_idx = min(len(words) - 1, max(1, len(words) // 2))

        # Subject tokens: words[0 : verb_idx]
        subj_tokens = tagged[0:verb_idx]
        # Object tokens: words[verb_idx + 1 :] (excluding prepositions)
        end_obj_idx = prep_indices[0] if prep_indices and prep_indices[0] > verb_idx else len(tagged)
        obj_tokens = tagged[verb_idx + 1 : end_obj_idx]

        # Parse Subject
        subject_info = self._extract_np(subj_tokens)
        
        # Parse Verb
        raw_verb = tagged[verb_idx][0].lower()
        verb_lemma = self._lemmatize_verb(raw_verb)

        # Parse Object
        object_info = self._extract_np(obj_tokens) if obj_tokens else None

        # Parse Prepositional Phrases
        prep_phrases = []
        for p_idx in prep_indices:
            p_word = tagged[p_idx][0].lower()
            p_rest = tagged[p_idx + 1 :]
            p_np = self._extract_np(p_rest)
            prep_phrases.append({
                "prep": p_word,
                "noun": p_np.get("noun", "thing")
            })

        return {
            "subject": subject_info,
            "verb": verb_lemma,
            "object": object_info,
            "prepositions": prep_phrases
        }

    def _extract_np(self, tokens: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Extracts quantifier, adjectives, and head noun from tokens."""
        if not tokens:
            return {"quantifier": "a", "noun": "entity", "adjectives": []}

        quantifier = "the"
        adjectives = []
        nouns = []

        for i, (w, tag) in enumerate(tokens):
            w_lower = w.lower()
            if i == 0 and w_lower in self.QUANTIFIER_MAP:
                quantifier = w_lower
            elif tag.startswith('JJ') or w_lower in ('financial', 'heavy', 'black', 'white', 'fierce', 'smart'):
                adjectives.append(w_lower)
            elif tag.startswith('NN') or i == len(tokens) - 1:
                nouns.append(w_lower)
            else:
                # Treat general content word as noun or adj
                nouns.append(w_lower)

        head_noun = nouns[-1] if nouns else "thing"
        is_relational = len(nouns) > 1 or head_noun in ('bank', 'edge', 'member', 'part')

        # If compound noun (e.g. "river bank"), add modifier
        if len(nouns) > 1:
            adjectives.extend(nouns[:-1])

        return {
            "quantifier": quantifier,
            "adjectives": adjectives,
            "noun": head_noun,
            "is_relational": is_relational
        }

    def _lemmatize_verb(self, verb: str) -> str:
        """Returns base lemma for common verbs."""
        v = verb.lower()
        if _NLTK_AVAILABLE and wn:
            lemma = wn.morphy(v, wn.VERB)
            if lemma:
                return lemma

        irregular_map = {
            'barks': 'bark', 'barking': 'bark', 'barked': 'bark',
            'flooded': 'flood', 'flooding': 'flood', 'floods': 'flood',
            'chased': 'chase', 'chases': 'chase', 'chasing': 'chase',
            'approved': 'approve', 'approves': 'approve', 'approving': 'approve',
            'is': 'be', 'was': 'be', 'are': 'be', 'were': 'be',
            'swims': 'swim', 'swimming': 'swim', 'swam': 'swim',
            'answers': 'answer', 'answered': 'answer', 'answering': 'answer'
        }
        return irregular_map.get(v, v)

    def convert_to_fol(self, mrs: MRS) -> Dict[str, Any]:
        """
        Converts an ERG-style MRS representation to a First-Order Logic (FOL) formula.

        Args:
            mrs: Input MRS object.

        Returns:
            Dict containing mathematical Unicode FOL, ASCII FOL, and step-by-step resolution.
        """
        # Separate relations by category
        quantifiers = []
        nominal_preds = {}   # var -> list of predicate names
        event_preds = []     # (pred_name, event_var, arg_vars)
        modifiers = {}       # var -> list of adj/modifier predicates

        for ep in mrs.rels:
            p = ep.pred
            if p.endswith('_q_rel'):
                quantifiers.append(ep)
            elif p.endswith('_n_1_rel') or p.endswith('_n_of_rel'):
                var = ep.arg0
                clean_p = re.sub(r'^_|_n_\d+_rel$|_n_of_rel$', '', p)
                nominal_preds.setdefault(var, []).append(clean_p)
            elif p.endswith('_v_1_rel') or p.endswith('_v_rel'):
                clean_p = re.sub(r'^_|_v_\d+_rel$|_v_rel$', '', p)
                args = [ep.args.get("ARG1", "x1")]
                if "ARG2" in ep.args:
                    args.append(ep.args["ARG2"])
                event_preds.append((clean_p, ep.arg0, args))
            elif p.endswith('_a_1_rel'):
                clean_p = re.sub(r'^_|_a_\d+_rel$', '', p)
                target_var = ep.args.get("ARG1", ep.arg0)
                modifiers.setdefault(target_var, []).append(clean_p)
            elif p.endswith('_p_rel'):
                clean_p = re.sub(r'^_|_p_rel$', '', p)
                arg1 = ep.args.get("ARG1", "e1")
                arg2 = ep.args.get("ARG2", "x2")
                event_preds.append((clean_p, ep.arg0, [arg1, arg2]))

        # Synthesize core proposition (Verb + arguments)
        prop_parts = []
        for pred_name, e_var, arg_list in event_preds:
            args_str = ", ".join(arg_list)
            prop_parts.append(f"{pred_name}({e_var}, {args_str})")

        core_event_fol = " ∧ ".join(prop_parts)
        core_event_ascii = " & ".join(prop_parts)

        # Scoping over quantifiers from right to left or according to quantifier type
        fol_unicode = core_event_fol
        fol_ascii = core_event_ascii

        steps = []
        steps.append(f"1. Core event relation: {core_event_fol}")

        # Bind event variable
        event_var = mrs.index
        fol_unicode = f"∃{event_var}. ({fol_unicode})"
        fol_ascii = f"exists {event_var}. ({fol_ascii})"
        steps.append(f"2. Event existential closure: {fol_unicode}")

        # Wrap quantifiers
        for q_ep in reversed(quantifiers):
            q_type = self.QUANTIFIER_MAP.get(q_ep.pred.replace('_rel', '').replace('_q', '').lstrip('_'), ('', 'existential'))[1]
            var = q_ep.arg0

            # Nominal restriction
            nouns = nominal_preds.get(var, [f"entity_{var}"])
            mods = modifiers.get(var, [])
            all_restr = [f"{m}({var})" for m in mods] + [f"{n}({var})" for n in nouns]
            restr_unicode = " ∧ ".join(all_restr)
            restr_ascii = " & ".join(all_restr)

            if q_type == 'universal':
                fol_unicode = f"∀{var}. ({restr_unicode} → {fol_unicode})"
                fol_ascii = f"all {var}. ({restr_ascii} -> {fol_ascii})"
                steps.append(f"3. Universal quantification over {var}: {fol_unicode}")
            elif q_type == 'negative':
                fol_unicode = f"¬∃{var}. ({restr_unicode} ∧ {fol_unicode})"
                fol_ascii = f"~exists {var}. ({restr_ascii} & {fol_ascii})"
                steps.append(f"3. Negative quantification over {var}: {fol_unicode}")
            else: # existential
                fol_unicode = f"∃{var}. ({restr_unicode} ∧ {fol_unicode})"
                fol_ascii = f"exists {var}. ({restr_ascii} & {fol_ascii})"
                steps.append(f"3. Existential quantification over {var}: {fol_unicode}")

        return {
            "surface_sentence": mrs.surface_text,
            "fol_formula": fol_unicode,
            "fol_ascii": fol_ascii,
            "mrs_index": mrs.index,
            "resolution_steps": steps
        }
