import os

chem_dir = 'app/ai_nutrition/chemometrics'
files = [f for f in os.listdir(chem_dir) if f.endswith('.py')]

total = 0
print('Chemometric Modules:')
print('='*60)

for f in sorted(files):
    path = os.path.join(chem_dir, f)
    with open(path, encoding='utf-8') as file:
        lines = len(file.readlines())
    print(f'{f}: {lines} lines')
    total += lines

print('='*60)
print(f'Total: {total} lines')
