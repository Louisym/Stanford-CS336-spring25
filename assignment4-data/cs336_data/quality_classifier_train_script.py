import fasttext
import os
import random

# File paths
HQ_FILE = '/Users/liuyimin/Projects/assignment4-data/data/clean_positive.txt'
LQ_FILE = '/Users/liuyimin/Projects/assignment4-data/data/clean_negative.txt'
COMBINED_FILE = '/Users/liuyimin/Projects/assignment4-data/data/quality_training_data.txt'
TRAIN_FILE = '/Users/liuyimin/Projects/assignment4-data/data/quality_train.txt'
VAL_FILE = '/Users/liuyimin/Projects/assignment4-data/data/quality_val.txt'
MODEL_FILE = '/Users/liuyimin/Projects/assignment4-data/data/quality_classifier.bin'

def combine_data():
    """Combine HQ and LQ data into a single training file"""
    print("Combining HQ and LQ data...")
    
    all_samples = []
    
    # Read HQ data
    if os.path.exists(HQ_FILE):
        with open(HQ_FILE, 'r', encoding='utf-8') as f:
            hq_lines = f.readlines()
            print(f"Loaded {len(hq_lines)} HQ samples")
            all_samples.extend(hq_lines)
    else:
        print(f"Warning: {HQ_FILE} not found!")
    
    # Read LQ data
    if os.path.exists(LQ_FILE):
        with open(LQ_FILE, 'r', encoding='utf-8') as f:
            lq_lines = f.readlines()
            print(f"Loaded {len(lq_lines)} LQ samples")
            all_samples.extend(lq_lines)
    else:
        print(f"Warning: {LQ_FILE} not found!")
    
    # Shuffle the combined data
    random.shuffle(all_samples)
    
    # Write combined data
    with open(COMBINED_FILE, 'w', encoding='utf-8') as f:
        f.writelines(all_samples)
    
    print(f"Combined {len(all_samples)} samples written to {COMBINED_FILE}")
    return len(all_samples)

def split_data(test_size=0.2):
    """Split data into training and validation sets"""
    print("Splitting data into train/validation sets...")
    
    with open(COMBINED_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Separate HQ and LQ lines for balanced splitting
    hq_lines = [line for line in lines if line.startswith('__label__HQ')]
    lq_lines = [line for line in lines if line.startswith('__label__LQ')]
    
    # Shuffle each group separately
    random.shuffle(hq_lines)
    random.shuffle(lq_lines)
    
    # Split each group
    hq_split = int(len(hq_lines) * (1 - test_size))
    lq_split = int(len(lq_lines) * (1 - test_size))
    
    train_lines = hq_lines[:hq_split] + lq_lines[:lq_split]
    val_lines = hq_lines[hq_split:] + lq_lines[lq_split:]
    
    # Shuffle the final datasets
    random.shuffle(train_lines)
    random.shuffle(val_lines)
    
    # Write training data
    with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    
    # Write validation data
    with open(VAL_FILE, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)
    
    print(f"Training set: {len(train_lines)} samples")
    print(f"Validation set: {len(val_lines)} samples")
    
    return len(train_lines), len(val_lines)

def train_classifier():
    """Train fastText quality classifier"""
    print("Training fastText quality classifier...")
    
    # Train the model
    model = fasttext.train_supervised(
        input=TRAIN_FILE,
        epoch=25,
        lr=0.1,
        wordNgrams=2,
        dim=100,
        ws=5,
        minCount=1,
        verbose=2
    )
    
    # Save the model
    model.save_model(MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")
    
    return model

def evaluate_model(model):
    """Evaluate the trained model on validation set"""
    print("Evaluating model on validation set...")
    
    # Test on validation set
    result = model.test(VAL_FILE)
    
    print(f"Validation samples: {result[0]}")
    print(f"Precision: {result[1]:.4f}")
    print(f"Recall: {result[2]:.4f}")
    
    # Test on some individual samples
    print("\n--- Sample Predictions ---")
    with open(VAL_FILE, 'r', encoding='utf-8') as f:
        sample_lines = f.readlines()[:10]  # First 10 validation samples
    
    for i, line in enumerate(sample_lines):
        true_label = line.split()[0]
        text = ' '.join(line.split()[1:]).strip()
        
        # Predict
        pred_labels, confidences = model.predict(text, k=1)
        pred_label = pred_labels[0]
        confidence = confidences[0]
        
        print(f"Sample {i+1}:")
        print(f"  True: {true_label}, Predicted: {pred_label}, Confidence: {confidence:.4f}")
        print(f"  Text preview: {text[:100]}...")
        print()

def create_quality_inference_function():
    """Create a quality scoring function for use in adapters"""
    inference_code = '''
def run_classify_quality(text: str) -> tuple[str, float]:
    """
    Classify text quality using the trained fastText model.
    
    Args:
        text: Input text to classify
        
    Returns:
        tuple: (label, confidence_score) where label is 'high-quality' or 'low-quality'
               and confidence_score is between 0 and 1
    """
    import fasttext
    
    # Load the model (you may want to cache this in a real application)
    model_path = '/Users/liuyimin/Projects/assignment4-data/data/quality_classifier.bin'
    model = fasttext.load_model(model_path)
    
    # Clean the text for prediction (remove newlines, extra spaces)
    cleaned_text = ' '.join(text.split())
    
    # Get prediction
    labels, confidences = model.predict(cleaned_text, k=1)
    
    # Convert fasttext label to our format
    fasttext_label = labels[0]
    confidence = float(confidences[0])
    
    if fasttext_label == '__label__HQ':
        return ('high-quality', confidence)
    else:  # __label__LQ
        return ('low-quality', confidence)
'''
    
    # Save to a separate file for easy import
    with open('/Users/liuyimin/Projects/assignment4-data/cs336_data/quality_inference.py', 'w') as f:
        f.write(inference_code)
    
    print("Quality inference function saved to cs336_data/quality_inference.py")
    print("You can add this function to your tests/adapters.py file")

def main():
    """Main training pipeline"""
    print("=== Quality Classifier Training Pipeline ===")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Step 1: Combine data
    total_samples = combine_data()
    
    if total_samples == 0:
        print("Error: No training data found!")
        return
    
    # Step 2: Split data
    train_size, val_size = split_data()
    
    # Step 3: Train classifier
    model = train_classifier()
    
    # Step 4: Evaluate model
    evaluate_model(model)
    
    # Step 5: Create inference function
    create_quality_inference_function()
    
    print("\n=== Training Complete! ===")
    print(f"Model saved at: {MODEL_FILE}")
    print("Use the run_classify_quality function for inference.")

if __name__ == '__main__':
    main()