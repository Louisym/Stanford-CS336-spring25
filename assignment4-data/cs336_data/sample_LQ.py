from tests.adapters import *
from fastwarc import ArchiveIterator, WarcRecordType
import random

WARC_FILE = '/Users/liuyimin/Projects/assignment4-data/example.warc.gz'
OUT_FILE = '/Users/liuyimin/Projects/assignment4-data/data/clean_negative.txt'
TARGET_SAMPLES = 793  # Match the number of HQ samples

def sample_LQ_process():
    # First pass: collect all potential records
    all_records = []
    
    print("First pass: collecting all potential LQ records...")
    with open(WARC_FILE, 'rb') as f:
        for record_idx, record in enumerate(ArchiveIterator(f)):
            if record.record_type == WarcRecordType.response:
                try:
                    raw_text = run_extract_text_from_html_bytes(record.reader.read())
                    if raw_text:
                        # Apply the same masking pipeline
                        emails_masked_text, emails_masked_cnt = run_mask_emails(raw_text)
                        phone_numbers_masked_text, phone_numbers_masked_cnt = run_mask_phone_numbers(emails_masked_text)
                        ips_masked_text, ips_masked_cnt = run_mask_ips(phone_numbers_masked_text)
                        
                        # Check if it passes Gopher quality filter
                        gopher_passed = run_gopher_quality_filter(ips_masked_text)
                        
                        # For LQ, we want records that either:
                        # 1. Don't pass the gopher filter (natural LQ)
                        # 2. Pass the filter but we'll label as LQ anyway for balanced training
                        all_records.append((ips_masked_text, gopher_passed, record_idx))
                        
                        if len(all_records) % 1000 == 0:
                            print(f"Collected {len(all_records)} potential records...")
                            
                except Exception as e:
                    print(f"Error processing record {record_idx}: {e}")
                    continue
    
    print(f"Total potential records collected: {len(all_records)}")
    
    # Prioritize records that don't pass gopher filter for more realistic LQ distribution
    failed_gopher = [r for r in all_records if not r[1]]  # Records that failed gopher filter
    passed_gopher = [r for r in all_records if r[1]]     # Records that passed gopher filter
    
    print(f"Records that failed Gopher filter: {len(failed_gopher)}")
    print(f"Records that passed Gopher filter: {len(passed_gopher)}")
    
    # Sample strategy: prefer failed gopher records, supplement with random passed records if needed
    selected_records = []
    
    if len(failed_gopher) >= TARGET_SAMPLES:
        # We have enough failed gopher records
        selected_records = random.sample(failed_gopher, TARGET_SAMPLES)
        print(f"Selected {TARGET_SAMPLES} records from failed Gopher filter records")
    else:
        # Use all failed gopher records + random sample from passed records
        selected_records = failed_gopher.copy()
        remaining_needed = TARGET_SAMPLES - len(failed_gopher)
        
        if len(passed_gopher) >= remaining_needed:
            selected_records.extend(random.sample(passed_gopher, remaining_needed))
            print(f"Selected {len(failed_gopher)} failed + {remaining_needed} passed Gopher records")
        else:
            # Not enough records total
            selected_records.extend(passed_gopher)
            print(f"WARNING: Only found {len(selected_records)} total records, needed {TARGET_SAMPLES}")
    
    # Shuffle the selected records
    random.shuffle(selected_records)
    
    # Write to output file
    print(f"Writing {len(selected_records)} records to {OUT_FILE}")
    with open(OUT_FILE, 'w') as out:
        for i, (text, gopher_passed, record_idx) in enumerate(selected_records):
            out.write('__label__LQ ' + text + '\n')
            if (i + 1) % 100 == 0:
                print(f"Written {i + 1}/{len(selected_records)} records...")
    
    print(f"Successfully created LQ dataset with {len(selected_records)} samples")
    
    # Print some statistics
    failed_count = sum(1 for _, gopher_passed, _ in selected_records if not gopher_passed)
    passed_count = len(selected_records) - failed_count
    print(f"Final composition: {failed_count} failed Gopher, {passed_count} passed Gopher")

def main():
    # Set random seed for reproducibility
    random.seed(42)
    sample_LQ_process()

if __name__ == '__main__':
    main()