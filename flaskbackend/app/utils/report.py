import os, uuid
from typing import List, Dict, Any
import pandas as pd

REPORT_DIR = os.environ.get('REPORT_DIR', 'reports')
os.makedirs(REPORT_DIR, exist_ok=True)

def save_report_html(plan: List[Dict[str,Any]], meta: Dict[str,Any]=None):
    df = pd.DataFrame(plan)
    filename = f'tumorheal_report_{uuid.uuid4().hex[:8]}.html'
    path = os.path.join(REPORT_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('<html><body>')
        f.write('<h2>TumorHeal Plan Report</h2>')
        if meta:
            f.write(f'<p>Meta: {meta}</p>')
        f.write(df.to_html(index=False))
        f.write('</body></html>')
    return path
