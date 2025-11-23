from tests.adapters import run_extract_text_from_html_bytes, run_identify_language
from fastwarc.warc import ArchiveIterator, WarcRecordType
import random

all_texts = []
with open('/Users/liuyimin/Projects/assignment4-data/example.warc.gz', 'rb') as f:
    for record in ArchiveIterator(f):
        if record.record_type == WarcRecordType.response:
            html_bytes = record.reader.read()
            try:
                clean_text = run_extract_text_from_html_bytes(html_bytes)
                if clean_text and len(clean_text.strip()) > 50:  # 过滤空文本和太短的文本
                    all_texts.append(clean_text)
            except Exception as e:
                print(f"Error processing record: {e}")
                continue

print(f'num of text chunks is {len(all_texts)}')

#initialize random sample 
random.seed(42)
sample_size = min(20, len(all_texts))
samples = random.sample(all_texts, sample_size)

classifier_res = []
for i, text in enumerate(samples):
    res_label, res_conf = run_identify_language(text)
    classifier_res.append({
          'sample_id': i+1,
          'predicted_lang': res_label,
          'confidence': res_conf,
          'text_preview': text[:200] + "..." if len(text) > 200 else text
      })

with open('sampled_language_res.txt', 'w', encoding='utf-8') as f:
    for result in classifier_res:
          f.write(f"=== Sample {result['sample_id']} ===\n")
          f.write(f"Predicted: {result['predicted_lang']} (confidence: {result['confidence']:.3f})\n")
          f.write(f"Text preview:\n{result['text_preview']}\n")
          f.write("-" * 50 + "\n\n")