import argparse
import glob
import os
import re
import sys
from datetime import datetime
from typing import List, Tuple, Optional


def extract_date_from_filename(filename: str) -> Optional[datetime]:
    """Extract date from filename containing YYYYMMDD pattern."""
    # Look for YYYYMMDD pattern in the entire filepath, not just basename
    date_match = re.search(r'(\d{8})', filename)
    if not date_match:
        return None
    
    date_str = date_match.group(1)
    try:
        year = int(date_str[0:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        return datetime(year, month, day)
    except ValueError:
        return None


def format_friendly_date(dt: datetime) -> str:
    """Convert datetime to friendly format like 'Monday May 1st, 2025'."""
    day_name = dt.strftime('%A')
    month_name = dt.strftime('%B')
    day_num = dt.day
    year = dt.year
    
    # Add ordinal suffix
    if 10 <= day_num % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day_num % 10, 'th')
    
    return f"{day_name} {month_name} {day_num}{suffix}, {year}"


def cluster_glob(filepaths: List[str]) -> str:
    """Generate markdown from files with friendly date headers."""
    if not filepaths:
        return "No files provided"
    
    # Process files and extract dates
    file_data: List[Tuple[datetime, str, str]] = []
    
    for filepath in filepaths:
        if not os.path.isfile(filepath):
            print(f"Warning: File not found {filepath}. Skipping.", file=sys.stderr)
            continue
            
        date = extract_date_from_filename(filepath)
        if date is None:
            print(f"Warning: Could not extract date from filename {filepath}. Skipping.", file=sys.stderr)
            continue
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            file_data.append((date, filepath, content))
        except Exception as e:
            print(f"Warning: Could not read file {filepath}: {e}", file=sys.stderr)
            continue
    
    if not file_data:
        return "No valid files with extractable dates found"
    
    # Sort by date
    file_data.sort(key=lambda x: x[0])
    
    # Generate markdown
    lines = []
    for date, filepath, content in file_data:
        friendly_date = format_friendly_date(date)
        lines.append(f"# {friendly_date}")
        lines.append("")
        lines.append(content.strip())
        lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate markdown from files with friendly date headers."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="File paths (use shell globbing: ~/dir/2025*/ponder*.md)",
    )
    
    args = parser.parse_args()
    
    try:
        markdown = cluster_glob(args.files)
        print(markdown)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
