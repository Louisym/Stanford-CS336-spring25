#!/usr/bin/env python3

import fasttext
from tests.adapters import run_classify_quality

# Test the actual classifier behavior
def test_classifier_behavior():
    print("=== Testing Classifier Behavior ===\n")
    
    # Load the test files
    low_quality_cc_path = "tests/fixtures/low_quality_cc.txt"
    high_quality_wiki_path = "tests/fixtures/high_quality_wiki_reference.txt"
    
    with open(low_quality_cc_path) as f:
        low_quality_cc = f.read()
    
    with open(high_quality_wiki_path) as f:
        high_quality_wiki = f.read()
    
    # Test directly with fasttext model
    MODEL_PATH = '/Users/liuyimin/Projects/assignment4-data/data/quality_classifier.bin'
    classifier = fasttext.load_model(MODEL_PATH)
    
    print("=== Low Quality CC Text ===")
    print(f"Text preview: {low_quality_cc[:200]}...")
    cleaned_low = low_quality_cc.replace('\n', '').replace('\r', '')
    label_low, conf_low = classifier.predict(cleaned_low)
    print(f"FastText prediction: {label_low[0]} (confidence: {conf_low[0]:.4f})")
    
    # Test with adapter function
    adapter_pred, adapter_conf = run_classify_quality(low_quality_cc)
    print(f"Adapter prediction: {adapter_pred} (confidence: {adapter_conf:.4f})")
    print(f"Expected: cc\n")
    
    print("=== High Quality Wiki Text ===")
    print(f"Text preview: {high_quality_wiki[:200]}...")
    cleaned_high = high_quality_wiki.replace('\n', '').replace('\r', '')
    label_high, conf_high = classifier.predict(cleaned_high)
    print(f"FastText prediction: {label_high[0]} (confidence: {conf_high[0]:.4f})")
    
    # Test with adapter function
    adapter_pred2, adapter_conf2 = run_classify_quality(high_quality_wiki)
    print(f"Adapter prediction: {adapter_pred2} (confidence: {adapter_conf2:.4f})")
    print(f"Expected: wiki\n")
    
    # Check if the problem is in the mapping logic
    print("=== Analysis ===")
    print(f"Low quality text classified as: {label_low[0]}")
    print(f"High quality text classified as: {label_high[0]}")
    
    if label_low[0] == '__label__HQ':
        print("❌ Problem: Low quality text is being classified as HIGH quality!")
    else:
        print("✅ Low quality text correctly classified as low quality")
        
    if label_high[0] == '__label__HQ':
        print("✅ High quality text correctly classified as high quality")
    else:
        print("❌ Problem: High quality text is being classified as LOW quality!")

if __name__ == "__main__":
    test_classifier_behavior()