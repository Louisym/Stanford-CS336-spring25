#!/usr/bin/env python3
"""
Convert MATH dataset from 'problem'/'solution' format to 'question'/'answer' format.
"""

import json
import os
import shutil
from pathlib import Path


def convert_jsonl_file(input_path: str, output_path: str, backup: bool = True):
    """
    Convert a JSONL file from problem/solution to question/answer format.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        backup: If True, create a backup of the original file
    """
    # Create backup if requested
    if backup and os.path.exists(input_path):
        backup_path = input_path + '.backup'
        shutil.copy2(input_path, backup_path)
        print(f"Created backup: {backup_path}")

    converted_count = 0
    already_correct = 0

    # Read and convert
    with open(input_path, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for line_num, line in enumerate(lines, 1):
            try:
                item = json.loads(line.strip())

                # Check if conversion is needed
                needs_conversion = False

                # Convert problem -> question
                if 'problem' in item and 'question' not in item:
                    item['question'] = item.pop('problem')
                    needs_conversion = True

                # Convert solution -> answer
                if 'solution' in item and 'answer' not in item:
                    item['answer'] = item.pop('solution')
                    needs_conversion = True

                if needs_conversion:
                    converted_count += 1
                else:
                    already_correct += 1

                # Write converted line
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                # Write original line if parsing fails
                f_out.write(line)

    return converted_count, already_correct


def main():
    """Convert all MATH dataset files."""

    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'math'

    files_to_convert = [
        'train.jsonl',
        'test.jsonl',
    ]

    print("Converting MATH dataset files from 'problem'/'solution' to 'question'/'answer' format...")
    print("=" * 80)

    total_converted = 0
    total_already_correct = 0

    for filename in files_to_convert:
        input_path = data_dir / filename

        if not input_path.exists():
            print(f"⚠️  File not found: {input_path}")
            continue

        print(f"\nProcessing: {filename}")

        # Convert in place (backup is created automatically)
        temp_path = str(input_path) + '.tmp'
        converted, already_correct = convert_jsonl_file(
            str(input_path),
            temp_path,
            backup=True
        )

        # Replace original with converted file
        shutil.move(temp_path, str(input_path))

        print(f"  ✓ Converted: {converted} entries")
        print(f"  ✓ Already correct: {already_correct} entries")

        total_converted += converted
        total_already_correct += already_correct

    print("\n" + "=" * 80)
    print(f"✅ Conversion complete!")
    print(f"   Total converted: {total_converted}")
    print(f"   Total already correct: {total_already_correct}")
    print(f"\n💡 Backup files created with '.backup' extension")
    print(f"   You can delete them once you verify everything works.")


if __name__ == '__main__':
    main()
