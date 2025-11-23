
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
