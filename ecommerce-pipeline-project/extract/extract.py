import pandas as pd
import os
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from typing import List, Dict, Any

# Initialize rich console
console = Console()

def get_data_directory() -> str:
    """
    Get the path to the data directory.
    
    Returns:
        str: Path to the data directory
    
    Raises:
        FileNotFoundError: If the data directory doesn't exist
    """
    # Get the absolute path to the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"[red]Directory {data_dir} does not exist. "
            "Please create a 'data' folder in the main project directory.[/red]"
        )
    
    console.print(f"[green]Found data directory: {data_dir}[/green]")
    return data_dir

def analyze_csv_file(file_path: str) -> Dict[str, Any]:
    """
    Analyze a single CSV file and return its summary.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        Dict[str, Any]: Summary of the CSV file
    """
    try:
        df = pd.read_csv(file_path)
        return {
            'file_name': os.path.basename(file_path),
            'num_rows': len(df),
            'num_columns': len(df.columns)
        }
    except Exception as e:
        return {
            'file_name': os.path.basename(file_path),
            'error': str(e)
        }

def process_csv_files(data_dir: str) -> List[Dict[str, Any]]:
    """
    Process all CSV files in the data directory.
    
    Args:
        data_dir (str): Path to the data directory
        
    Returns:
        List[Dict[str, Any]]: List of summaries for each CSV file
    """
    summary = []
    
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(data_dir, file)
            console.print(f"[yellow]Processing file: {file}[/yellow]")
            summary.append(analyze_csv_file(file_path))
    
    return summary

def display_summary(summary: List[Dict[str, Any]]) -> None:
    """
    Display the summary in a formatted table using rich.
    
    Args:
        summary (List[Dict[str, Any]]): List of file summaries
    """
    table = Table(title="CSV Files Summary")
    
    # Add columns
    table.add_column("File Name", style="cyan")
    table.add_column("Number of Rows", style="green")
    table.add_column("Number of Columns", style="blue")
    table.add_column("Error", style="red")
    
    # Add rows
    for item in summary:
        error = item.get('error', '')
        if error:
            table.add_row(
                item['file_name'],
                '',
                '',
                error
            )
        else:
            table.add_row(
                item['file_name'],
                str(item['num_rows']),
                str(item['num_columns']),
                ''
            )
    
    console.print(table)

def main() -> None:
    """
    Main function to orchestrate the CSV analysis process.
    """
    try:
        # Get data directory
        data_dir = get_data_directory()
        
        # Process CSV files
        console.print("[bold blue]Starting CSV analysis...[/bold blue]")
        summary = process_csv_files(data_dir)
        
        # Display results
        display_summary(summary)
        
        console.print("[bold green]Analysis completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        raise

if __name__ == "__main__":
    main()
