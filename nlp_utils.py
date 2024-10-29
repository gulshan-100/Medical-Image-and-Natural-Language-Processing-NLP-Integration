import logging
from transformers import pipeline, AutoTokenizer, T5ForConditionalGeneration
import spacy
import re

logger = logging.getLogger(__name__)

# Initialize models globally
try:
    # Load T5 model for summarization - using correct model class
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    
    # Load SpaCy for medical terms
    nlp = spacy.load("en_core_sci_md")
    
    # Load classifier for categorization
    classifier = pipeline("zero-shot-classification")
    
    print("NLP models loaded successfully")
except Exception as e:
    print(f"Error loading NLP models: {str(e)}")
    raise

def clean_text(text):
    """Remove special characters and fix spacing"""
    # Remove special characters
    text = re.sub(r'[^\w\s.]', '', text)
    # Fix spacing
    return ' '.join(text.split())

def find_medical_terms(text):
    """Find medical terms in the text"""
    doc = nlp(text)
    
    # Initialize categories
    medical_terms = {
        "diseases": [],
        "symptoms": [],
        "medications": [],
        "procedures": []
    }
    
    # Categorize each medical term
    for entity in doc.ents:
        if entity.label_ in ["DISEASE", "SYNDROME"]:
            medical_terms["diseases"].append(entity.text)
        elif entity.label_ in ["SYMPTOM", "FINDING"]:
            medical_terms["symptoms"].append(entity.text)
        elif entity.label_ == "CHEMICAL":
            medical_terms["medications"].append(entity.text)
        elif entity.label_ == "PROCEDURE":
            medical_terms["procedures"].append(entity.text)
    
    return medical_terms

def create_summary(text):
    """Create a summary of the medical text"""
    try:
        # Prepare text for model
        inputs = tokenizer.encode(
            "summarize medical: " + text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        # Generate summary
        summary_ids = model.generate(
            inputs,
            max_length=150,
            min_length=40,
            num_beams=4,
            early_stopping=True
        )
        
        # Decode summary
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Error in create_summary: {str(e)}")
        return "Summary generation failed"

def sort_by_urgency(medical_terms):
    """Sort medical findings by urgency level"""
    try:
        categories = {
            "urgent": [],
            "important": [],
            "routine": []
        }
        
        # Combine all findings
        all_findings = medical_terms["diseases"] + medical_terms["symptoms"]
        
        # Categorize each finding
        for finding in all_findings:
            result = classifier(
                finding,
                candidate_labels=["urgent", "important", "routine"]
            )
            category = result['labels'][0]
            categories[category].append(finding)
        
        return categories
    except Exception as e:
        logger.error(f"Error in sort_by_urgency: {str(e)}")
        return {"urgent": [], "important": [], "routine": []}

def get_recommendations(medical_terms, urgency_categories):
    """Generate medical recommendations"""
    try:
        recommendations = []
        
        # Check for urgent cases
        if urgency_categories["urgent"]:
            recommendations.append("⚠️ IMMEDIATE MEDICAL ATTENTION REQUIRED")
        
        # Medication recommendations
        if medical_terms["medications"]:
            recommendations.append("💊 Continue prescribed medications as directed")
        
        # Symptom monitoring
        if medical_terms["symptoms"]:
            recommendations.append("📋 Monitor symptoms and maintain medical record")
        
        # General recommendations
        if medical_terms["diseases"]:
            recommendations.append("👨‍⚕️ Regular follow-up with healthcare provider recommended")
        
        return recommendations
    except Exception as e:
        logger.error(f"Error in get_recommendations: {str(e)}")
        return ["Unable to generate recommendations"]

def generate_medical_report(text):
    """Main function to generate complete medical report"""
    try:
        logger.debug("Starting report generation")
        if not text or not isinstance(text, str):
            raise ValueError("Invalid input text")

        # Preprocess text
        logger.debug("Preprocessing text")
        cleaned_text = clean_text(text)
        
        # Extract medical entities
        logger.debug("Extracting medical entities")
        entities = find_medical_terms(cleaned_text)
        
        # Generate summary
        logger.debug("Generating summary")
        summary = create_summary(cleaned_text)
        
        # Sort by urgency
        logger.debug("Categorizing findings")
        categories = sort_by_urgency(entities)
        
        # Generate recommendations
        logger.debug("Generating recommendations")
        recommendations = get_recommendations(entities, categories)
        
        logger.debug("Report generation completed successfully")
        return {
            "summary": summary,
            "medical_terms": entities,
            "urgency_levels": categories,
            "recommendations": recommendations
        }
        
    except Exception as e:
        logger.error(f"Error in report generation: {str(e)}")
        return {"error": f"Report generation failed: {str(e)}"}

# Example usage
if __name__ == "__main__":
    test_text = """
    Patient presents with severe headache and high fever of 102°F. 
    Currently taking acetaminophen for pain management. 
    History of migraines and seasonal allergies.
    """
    report = generate_medical_report(test_text)
    print(report)