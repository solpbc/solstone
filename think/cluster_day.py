import argparse
import os
import re
import sys
from datetime import datetime, timedelta
from collections import defaultdict

def process_files(date_str, folder_path):
    """
    Finds JSON files for a specific date in a folder, groups them by 5-minute
    intervals, and prints their content as Markdown.
    """
    # Regex to capture prefix, date, time, and optional suffix from filenames
    # Example: monitor_1_20250524_140241_diff.json
    # prefix: monitor_1
    # file_date_str: 20250524
    # time_str: 140241
    # suffix_part: _diff (this is captured by the fourth group)
    filename_pattern = re.compile(r"^(.*?)_(\d{8})_(\d{6})(.*?)\.json$")

    collected_files_data = []

    for filename in os.listdir(folder_path):
        match = filename_pattern.match(filename)
        if match:
            prefix, file_date_str_from_name, time_str, _ = match.groups()

            if file_date_str_from_name == date_str:
                try:
                    year = int(date_str[0:4])
                    month = int(date_str[4:6])
                    day = int(date_str[6:8])
                    
                    hour = int(time_str[0:2])
                    minute = int(time_str[2:4])
                    second = int(time_str[4:6])
                    
                    timestamp = datetime(year, month, day, hour, minute, second)
                    
                    full_path = os.path.join(folder_path, filename)
                    
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                    except Exception as e:
                        print(f"Warning: Could not read file {filename}: {e}", file=sys.stderr)
                        continue # Skip this file if unreadable
                    
                    collected_files_data.append({
                        'filepath': full_path,
                        'basename': filename,
                        'timestamp': timestamp,
                        'prefix': prefix,
                        'content': content
                    })
                except ValueError:
                    # This might happen if date/time components in filename are not valid integers
                    print(f"Warning: Could not parse date/time from filename {filename}. Skipping.", file=sys.stderr)
    
    # Sort all collected files by their precise timestamp
    collected_files_data.sort(key=lambda x: x['timestamp'])
    
    # Group files into 5-minute intervals
    grouped_files = defaultdict(list)
    for file_data in collected_files_data:
        ts = file_data['timestamp']
        # Calculate the start of the 5-minute interval
        interval_minute = ts.minute - (ts.minute % 5)
        interval_start_time = ts.replace(minute=interval_minute, second=0, microsecond=0)
        grouped_files[interval_start_time].append(file_data)
        
    # Print markdown output
    # Sort the 5-minute intervals chronologically
    sorted_interval_keys = sorted(grouped_files.keys())
    
    if not sorted_interval_keys:
        print(f"No JSON files found for date {date_str} matching the expected filename pattern in {folder_path}.")
        return

    for interval_start in sorted_interval_keys:
        interval_end = interval_start + timedelta(minutes=5)
        # Section header for the 5-minute period
        print(f"## {interval_start.strftime('%Y-%m-%d %H:%M')} - {interval_end.strftime('%H:%M')}")
        print() # Adds a blank line for better Markdown spacing
        
        files_in_group = grouped_files[interval_start]
        # Files within this group are already sorted by their full timestamp
        # because `collected_files_data` was sorted before grouping.
        for file_data in files_in_group:
            print(f"### {file_data['prefix']} ({file_data['basename']})")
            print("```json")
            # Strip content to remove potential leading/trailing whitespace from file read
            print(file_data['content'].strip()) 
            print("```")
            print() # Adds a blank line for better Markdown spacing

def main():
    parser = argparse.ArgumentParser(
        description="Organize JSON files by date and time into 5-minute intervals and print their content as Markdown. "
                    "Expects filenames like 'prefix_YYYYMMDD_HHMMSS_suffix.json' or 'prefix_YYYYMMDD_HHMMSS.json'."
    )
    parser.add_argument("date", 
                        help="The date to filter files by, in YYYYMMDD format (e.g., 20250524).")
    parser.add_argument("folder_path", 
                        help="The path to the folder containing the JSON files.")

    args = parser.parse_args()

    # Validate date format argument
    if not re.match(r"^\d{8}$", args.date):
        print("Error: Date argument format must be YYYYMMDD (e.g., 20250524).", file=sys.stderr)
        sys.exit(1)

    # Validate folder_path argument
    if not os.path.isdir(args.folder_path):
        print(f"Error: Folder not found at specified path: {args.folder_path}", file=sys.stderr)
        sys.exit(1)
        
    process_files(args.date, args.folder_path)

if __name__ == "__main__":
    main()