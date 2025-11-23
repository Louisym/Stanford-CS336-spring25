from tests.adapters import *
from fastwarc import ArchiveIterator, WarcRecordType
WARC_FILE = '/Users/liuyimin/Projects/assignment4-data/data/subsampled_positive_urls.warc.gz'
OUT_FILE = '/Users/liuyimin/Projects/assignment4-data/data/clean_positive.txt'

def clean_positive_process():
    cnt = 0
    with open(WARC_FILE, 'rb') as f, open(OUT_FILE, 'w') as out:
        for record in ArchiveIterator(f):
            if record.record_type == WarcRecordType.response:
                raw_text = run_extract_text_from_html_bytes(record.reader.read())
                if raw_text:
                    #We accept all kinds of languages here
                    #do mask below
                    emails_masked_text, emails_masked_cnt = run_mask_emails(raw_text)
                    print(f'mask {emails_masked_cnt} emails this record.\n')
                    phone_numbers_masked_text, phone_numbers_masked_cnt = run_mask_phone_numbers(emails_masked_text)
                    print(f'mask {phone_numbers_masked_cnt} phone this record.\n')
                    ips_masked_text, ips_masked_cnt = run_mask_ips(phone_numbers_masked_text)
                    print(f'mask {ips_masked_cnt} ips this record.\n')
                    #then, do gopher quality cleaning
                    gopher_text = run_gopher_quality_filter(ips_masked_text)
                    print(f'Gopher quality filter result: {gopher_text}')
                    if gopher_text:
                        #if it matches gopher quality, we write into the outputfile
                        out.write('__label__HQ ' + ips_masked_text + '\n')
                        cnt += 1
                        print(f'Wrote record {cnt} to output file.')
                    else:
                        print(f'Record failed quality filter. Text length: {len(ips_masked_text)} chars')
                else:
                    print('Failed to extract text from HTML.')
            print(f'Total records written so far: {cnt}')
def main():
    clean_positive_process()

if __name__ == '__main__':
    main()



