import os
from pathlib import Path

def count_lines_in_directory(directory):
    total_lines = 0
    file_count = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = len(f.readlines())
                        total_lines += lines
                        file_count += 1
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
    
    return total_lines, file_count

# Count lines in ai_nutrition
ai_nutrition_path = r'c:\Users\Codeternal\Music\wellomex\flaskbackend\app\ai_nutrition'
total, count = count_lines_in_directory(ai_nutrition_path)

print(f"Total Python files: {count}")
print(f"Total lines of code: {total:,}")
