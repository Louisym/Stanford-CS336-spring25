from tests.adapters import run_extract_text_from_html_bytes
from fastwarc.warc import ArchiveIterator, WarcRecordType

# 处理WARC文件
warc_extracts = []
with open('/Users/liuyimin/Projects/assignment4-data/example.warc.gz', 'rb') as f:
    for i, record in enumerate(ArchiveIterator(f)):
        if record.record_type == WarcRecordType.response:
            html_bytes = record.reader.read()
            clean_text = run_extract_text_from_html_bytes(html_bytes)
            if clean_text:  # 添加空值检查
                warc_extracts.append(clean_text)

            # 只处理前几个记录用于比较
            if len(warc_extracts) >= 5:
                break

# 处理对应的WET文件
wet_extracts = []
with open('/Users/liuyimin/Projects/assignment4-data/example.warc.wet.gz', 'rb') as f:
    for i, record in enumerate(ArchiveIterator(f)):
        if record.record_type == WarcRecordType.conversion:
            wet_text = record.reader.read().decode('utf-8')
            wet_extracts.append(wet_text)

            if len(wet_extracts) >= 5:
                break

# 比较并保存结果
with open('comparison_results.txt', 'w', encoding='utf-8') as f:
    for i, (warc_text, wet_text) in enumerate(zip(warc_extracts, wet_extracts)):
        f.write(f"=== Record {i+1} ===\n")
        f.write("WARC Extraction:\n")
        f.write(warc_text[:500] + "...\n\n")  # 只显示前500字符
        f.write("WET Extraction:\n")
        f.write(wet_text[:500] + "...\n\n")
        f.write("-" * 50 + "\n\n")