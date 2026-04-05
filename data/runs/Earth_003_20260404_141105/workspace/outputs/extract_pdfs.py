from pathlib import Path
outdir = Path('/outputs/pdf_text')
outdir.mkdir(parents=True, exist_ok=True)
files = sorted(Path('related_work').glob('*.pdf'))
print('PDFS', [f.name for f in files])
for pdf in files:
    text = None
    errs = []
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf))
        text = '\n'.join(page.extract_text() or '' for page in reader.pages)
    except Exception as e:
        errs.append(f'pypdf:{e}')
    if not text or len(text.strip()) < 200:
        try:
            import fitz
            doc = fitz.open(str(pdf))
            text = '\n'.join(page.get_text() for page in doc)
        except Exception as e:
            errs.append(f'fitz:{e}')
    out = outdir / (pdf.stem + '.txt')
    out.write_text(text or '')
    print(pdf.name, 'chars', len(text or ''), 'errs', errs)
