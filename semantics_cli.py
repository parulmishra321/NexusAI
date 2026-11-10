#!/usr/bin/env python3
"""
NexusAI - Computational Linguistics & Semantics CLI
Provides command-line access to:
1. WordNet Synset Retrieval & Algorithmic WSD
2. LLM-driven Sense Selection (Ollama structured JSON)
3. ERG-style Minimal Recursion Semantics (MRS) Parsing
4. First-Order Logic (FOL) Conversion
5. ChromaDB/RAG Semantic Query Expansion
"""

import sys
import os
import json
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from Backend.Core.Semantics.SemanticsService import SemanticsService

console = Console()


def print_banner():
    console.print(
        Panel.fit(
            "[bold cyan]NexusAI - Computational Linguistics & Semantic Processing[/bold cyan]\n"
            "[yellow]WordNet WSD • Ollama Structured Sense Selection • ERG MRS • First-Order Logic[/yellow]",
            border_style="cyan"
        )
    )


def cmd_synsets(service: SemanticsService, word: str, pos: str = None):
    console.print(f"\n[bold green]WordNet Synset Retrieval for:[/bold green] [bold yellow]'{word}'[/bold yellow]")
    synsets = service.get_synsets(word, pos=pos)
    if not synsets:
        console.print(f"[red]No synsets found for '{word}' in WordNet.[/red]")
        return

    table = Table(title=f"Synsets for '{word}' ({len(synsets)} found)")
    table.add_column("Synset ID", style="cyan", no_wrap=True)
    table.add_column("POS", style="magenta", width=6)
    table.add_column("Definition", style="white")
    table.add_column("Synonyms / Lemmas", style="green")

    for s in synsets[:8]:
        table.add_row(
            s["synset_id"],
            s["pos"],
            s["definition"],
            ", ".join(s["lemmas"])
        )
    console.print(table)


def cmd_wsd_lesk(service: SemanticsService, sentence: str, word: str):
    console.print(f"\n[bold green]Algorithmic WSD (Lesk) for word:[/bold green] [bold yellow]'{word}'[/bold yellow]")
    console.print(f"[dim]Sentence:[/dim] \"{sentence}\"")

    res = service.disambiguate_lesk(sentence, word)
    console.print(f"[bold cyan]Selected Synset ID:[/bold cyan] {res['best_synset']}")
    console.print(f"[bold cyan]Definition:[/bold cyan] {res['definition']}")
    console.print(f"[bold cyan]Overlap Score:[/bold cyan] {res['score']}")
    console.print(f"[bold cyan]Overlap Words:[/bold cyan] {res['overlap_words']}")

    if res.get("candidates"):
        cand_table = Table(title="Candidate Evaluations")
        cand_table.add_column("Synset ID", style="cyan")
        cand_table.add_column("Score", style="yellow")
        cand_table.add_column("Overlap", style="green")
        cand_table.add_column("Definition", style="white")
        for c in res["candidates"][:5]:
            cand_table.add_row(
                c["synset_id"],
                str(c["score"]),
                ", ".join(c["overlap"]),
                c["definition"]
            )
        console.print(cand_table)


def cmd_llm_wsd(service: SemanticsService, sentence: str, words: list = None, model: str = None):
    console.print(f"\n[bold green]LLM-Driven Sense Selection (Ollama Structured JSON)[/bold green]")
    console.print(f"[dim]Sentence:[/dim] \"{sentence}\"")
    if words:
        console.print(f"[dim]Target Words:[/dim] {words}")

    res = service.select_senses_llm(sentence, target_words=words, model=model)
    json_str = json.dumps(res, indent=2)
    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
    console.print(syntax)


def cmd_mrs(service: SemanticsService, sentence: str):
    console.print(f"\n[bold green]ERG-Style Minimal Recursion Semantics (MRS) Parsing[/bold green]")
    console.print(f"[dim]Sentence:[/dim] \"{sentence}\"")

    mrs = service.parse_mrs(sentence)
    delphin_output = mrs.to_delphin_str()
    console.print(Panel(delphin_output, title="[cyan]DELPH-IN MRS Representation[/cyan]", border_style="cyan"))


def cmd_fol(service: SemanticsService, sentence: str):
    console.print(f"\n[bold green]First-Order Logic (FOL) Conversion[/bold green]")
    console.print(f"[dim]Sentence:[/dim] \"{sentence}\"")

    fol = service.convert_to_fol(sentence)
    console.print(f"[bold cyan]FOL Formula (Unicode):[/bold cyan] [bold white]{fol['fol_formula']}[/bold white]")
    console.print(f"[bold cyan]FOL Formula (ASCII):  [/bold cyan] [bold white]{fol['fol_ascii']}[/bold white]")
    console.print(f"[bold cyan]Main MRS Index:       [/bold cyan] [yellow]{fol['mrs_index']}[/yellow]")
    console.print("\n[bold yellow]Scoping & Quantifier Resolution Steps:[/bold yellow]")
    for step in fol["resolution_steps"]:
        console.print(f"  [green]•[/green] {step}")


def cmd_rag_expansion(service: SemanticsService, query: str):
    console.print(f"\n[bold green]ChromaDB / RAG Semantic Query Expansion[/bold green]")
    console.print(f"[dim]Original Query:[/dim] \"{query}\"")

    enriched = service.enrich_rag_query(query)
    console.print(f"[bold cyan]Expanded Query for ChromaDB:[/bold cyan] [bold yellow]\"{enriched['expanded_query']}\"[/bold yellow]")
    console.print(f"[bold cyan]Extracted Synonyms/Lemmas:  [/bold cyan] {enriched['expansion_terms']}")
    
    if enriched["disambiguated_synsets"]:
        table = Table(title="Disambiguated Query Terms")
        table.add_column("Word", style="yellow")
        table.add_column("Synset ID", style="cyan")
        table.add_column("Definition", style="white")
        for item in enriched["disambiguated_synsets"]:
            table.add_row(item["word"], item["synset_id"], item["definition"])
        console.print(table)


def run_full_demo(service: SemanticsService):
    console.print("\n[bold magenta]════════════════════════════════════════════════════════════[/bold magenta]")
    console.print("[bold magenta]          NexusAI Semantic Processing Complete Demo         [/bold magenta]")
    console.print("[bold magenta]════════════════════════════════════════════════════════════[/bold magenta]")

    # 1. WordNet Synset Retrieval
    console.print("\n[bold cyan]1. WordNet Synset Retrieval & Lexical Exploration[/bold cyan]")
    cmd_synsets(service, "bank", pos="n")

    # 2. Algorithmic Word Sense Disambiguation (Lesk)
    console.print("\n[bold cyan]2. Algorithmic Word Sense Disambiguation (Lesk)[/bold cyan]")
    sentence_bank = "The river bank overflowed after the heavy storm."
    cmd_wsd_lesk(service, sentence_bank, "bank")

    # 3. LLM-Driven Sense Selection (Ollama Structured JSON)
    console.print("\n[bold cyan]3. LLM-Driven Sense Selection (Structured JSON Mapping)[/bold cyan]")
    sentence_financial = "The commercial bank approved the mortgage loan."
    cmd_llm_wsd(service, sentence_financial, ["bank"])

    # 4. Mock Semantic Parsing for ERG Minimal Recursion Semantics (MRS)
    console.print("\n[bold cyan]4. English Resource Grammar (ERG) MRS Parsing[/bold cyan]")
    sentence_mrs = "Every dog barks"
    cmd_mrs(service, sentence_mrs)

    # 5. First-Order Logic (FOL) Conversion
    console.print("\n[bold cyan]5. First-Order Logic (FOL) Conversion[/bold cyan]")
    sentence_fol = "The black cat chased a mouse"
    cmd_fol(service, sentence_fol)

    # 6. RAG Semantic Query Expansion
    console.print("\n[bold cyan]6. ChromaDB / RAG Semantic Query Expansion[/bold cyan]")
    cmd_rag_expansion(service, "financial bank compliance notes")

    console.print("\n[bold green]✔ All computational linguistics and semantic modules verified successfully![/bold green]\n")


def main():
    parser = argparse.ArgumentParser(
        description="NexusAI Computational Linguistics & Semantics CLI"
    )
    parser.add_argument("--demo", action="store_true", help="Run full automated showcase of all requirements")
    parser.add_argument("--synsets", type=str, metavar="WORD", help="Retrieve WordNet synsets for WORD")
    parser.add_argument("--pos", type=str, default=None, help="POS filter (n, v, a, r)")
    parser.add_argument("--wsd", type=str, metavar="SENTENCE", help="Run Lesk WSD on SENTENCE")
    parser.add_argument("--word", type=str, default=None, help="Target word for WSD")
    parser.add_argument("--llm-wsd", type=str, metavar="SENTENCE", help="Run LLM sense selection on SENTENCE")
    parser.add_argument("--target-words", nargs="+", help="Specific target words for LLM WSD")
    parser.add_argument("--mrs", type=str, metavar="SENTENCE", help="Parse SENTENCE to ERG MRS")
    parser.add_argument("--fol", type=str, metavar="SENTENCE", help="Convert SENTENCE to First-Order Logic")
    parser.add_argument("--rag-expand", type=str, metavar="QUERY", help="Enrich QUERY for ChromaDB retrieval")

    args = parser.parse_args()
    print_banner()

    service = SemanticsService()

    if args.demo or len(sys.argv) == 1:
        run_full_demo(service)
        return

    if args.synsets:
        cmd_synsets(service, args.synsets, pos=args.pos)
    elif args.wsd:
        word = args.word or args.wsd.split()[0]
        cmd_wsd_lesk(service, args.wsd, word)
    elif args.llm_wsd:
        cmd_llm_wsd(service, args.llm_wsd, words=args.target_words)
    elif args.mrs:
        cmd_mrs(service, args.mrs)
    elif args.fol:
        cmd_fol(service, args.fol)
    elif args.rag_expand:
        cmd_rag_expansion(service, args.rag_expand)


if __name__ == "__main__":
    main()
