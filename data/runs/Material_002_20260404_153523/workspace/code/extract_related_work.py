from pathlib import Path
from PyPDF2 import PdfReader

out_dir = Path('outputs/related_work_text')
out_dir.mkdir(parents=True, exist_ok=True)
for pdf_path in sorted(Path('related_work').glob('*.pdf')):
    reader = PdfReader(str(pdf_path))
    texts = []
    for i, page in enumerate(reader.pages):
        try:
            texts.append(page.extract_text() or '')
        except Exception as e:
            texts.append(f'[[EXTRACTION_ERROR page={i}: {e}]]')
    out_path = out_dir / (pdf_path.stem + '.txt')
    out_path.write_text('\n\n'.join(texts), encoding='utf-8')
    print(f'Wrote {out_path}')
