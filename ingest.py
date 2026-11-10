import os
import sys

# Ensure backend logic is importable
current_dir = os.getcwd()
sys.path.append(current_dir)

from Backend.Core.Memory.Ingestion import IngestionPipeline
from rich.console import Console

console = Console()

def main():
    console.print("[bold blue]Nexus AI - Document Ingestion[/bold blue]")
    
    pipeline = IngestionPipeline()
    docs_path = pipeline.docs_path
    
    console.print(f"Scanning directory: [yellow]{docs_path}[/yellow]")
    
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)
        console.print("[red]Directory not found. Created it.[/red]")
        console.print("Please copy .txt files into this folder and run again.")
        return

    try:
        count = pipeline.ingest_all()
        if count == 0:
            console.print("[yellow]No supported files found or indexed.[/yellow]")
        else:
            console.print(f"[green]Successfully indexed {count} documents![/green]")
            
    except Exception as e:
        console.print(f"[bold red]Error during ingestion:[/bold red] {e}")

if __name__ == "__main__":
    main()
