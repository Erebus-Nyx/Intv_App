"""
Enhanced dynamic module processor that uses the new DynamicHybridProcessor
for universal intelligent analysis followed by policy-structured output.

This replaces the old dynamic_module.py approach with the new scalable hybrid system.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Import the new dynamic hybrid processor
from .dynamic_hybrid_processor import DynamicHybridProcessor

def enhanced_dynamic_module_output(
    text_content: str,
    module_key: Optional[str] = None,
    output_path: Optional[str] = None,
    metadata: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Enhanced module processing using DynamicHybridProcessor
    
    Args:
        text_content: The text content to analyze
        module_key: Optional module type (adult, child, casefile, etc.)
        output_path: Optional path to save output
        metadata: Optional metadata about the content
        
    Returns:
        Dict containing analysis results and policy-structured output
    """
    try:
        # Initialize the dynamic hybrid processor
        processor = DynamicHybridProcessor()
        
        # Process the content
        result = processor.process(
            text_content=text_content,
            module_type=module_key,
            metadata=metadata
        )
        
        # Convert to the expected output format
        output = _convert_to_legacy_format(result, text_content)
        
        # Save to file if requested
        if output_path:
            _save_output(output, output_path, result["module"])
        
        return output
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "success": False,
            "is_final": False
        }

def _convert_to_legacy_format(hybrid_result: Dict, original_text: str) -> Dict[str, Any]:
    """
    Convert DynamicHybridProcessor output to legacy format expected by pipeline
    """
    policy_output = hybrid_result.get("policy_structured_output", {})
    generic_analysis = hybrid_result.get("generic_analysis", {})
    
    # Build narrative from policy-structured variables
    narrative_lines = []
    pending_questions = []
    
    for var_name, var_data in policy_output.items():
        value = var_data.get("value", "")
        default = var_data.get("default", "")
        hint = var_data.get("hint", "")
        confidence = var_data.get("confidence", 0.0)
        
        # Clean up the value
        if isinstance(value, str):
            value = value.replace("{", "").replace("}", "")
            value = value.replace("\\n", "\n").replace("<br>", "\n")
        
        # Check if we need clarification
        if not value or value == default:
            if confidence < 0.3:  # Low confidence threshold
                pending_questions.append(f"Please provide a value for {var_name} ({hint})")
        
        # Add to narrative if we have a meaningful value
        if value and value != default:
            # Use hint as label, fallback to variable name
            label = hint if hint else var_name.replace("_", " ").title()
            narrative_lines.append(f"{label}: {value}")
        elif default and default != "[No Name]" and default != "":
            # Use default if it's meaningful
            label = hint if hint else var_name.replace("_", " ").title()
            narrative_lines.append(f"{label}: {default}")
    
    narrative = "\n".join(narrative_lines)
    
    # Add generic analysis summary if narrative is sparse
    if len(narrative_lines) < 3:
        extracted_data = generic_analysis.get("extracted_data", {})
        
        # Add key extracted information
        if "personal_info" in extracted_data:
            personal = extracted_data["personal_info"]
            if personal.get("names"):
                narrative += f"\nIdentified Individual: {personal['names'][0]}"
        
        if "behavioral_observations" in extracted_data:
            behavioral = extracted_data["behavioral_observations"]
            if behavioral.get("summary"):
                narrative += f"\nObservations: {behavioral['summary']}"
        
        if "family_structure" in extracted_data:
            family = extracted_data["family_structure"]
            if family.get("summary"):
                narrative += f"\nFamily Information: {family['summary']}"
    
    # Determine status
    clarification_needed = len(pending_questions) > 0
    is_final = not clarification_needed
    
    result = {
        "status": "pending" if clarification_needed else "success",
        "success": not clarification_needed,
        "narrative": narrative,
        "clarification_needed": clarification_needed,
        "pending_questions": pending_questions,
        "is_final": is_final,
        
        # Additional data for advanced users
        "hybrid_analysis": {
            "module_type": hybrid_result.get("module"),
            "confidence_score": hybrid_result.get("confidence_score"),
            "processing_approach": hybrid_result.get("processing_approach"),
            "analysis_timestamp": hybrid_result.get("analysis_timestamp"),
            "auto_detected": hybrid_result.get("auto_detected", False)
        },
        
        # Raw analysis data
        "raw_analysis": {
            "generic_analysis": generic_analysis,
            "policy_structured_output": policy_output,
            "content_statistics": generic_analysis.get("content_statistics", {})
        }
    }
    
    return result

def _save_output(output: Dict, output_path: str, module_type: str) -> None:
    """Save output to file and cache"""
    try:
        # Save to specified path
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        # Also save to cache directory
        cache_dir = Path(__file__).parent.parent.parent / '.cache'
        cache_dir.mkdir(exist_ok=True)
        
        # Generate cache filename
        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H%M')
        
        # Try to extract name from narrative
        narrative = output.get("narrative", "")
        name = "noname"
        for line in narrative.split('\n'):
            if "name:" in line.lower() or "individual:" in line.lower():
                parts = line.split(':', 1)
                if len(parts) > 1:
                    name = parts[1].strip().replace(' ', '_')
                    name = ''.join(c for c in name if c.isalnum() or c in '_-')[:20]
                    break
        
        cache_filename = f"{date_str}_{time_str}_{module_type}_{name}.json"
        cache_path = cache_dir / cache_filename
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
            
        print(f"[INFO] Output saved to {output_path}")
        print(f"[INFO] Output cached to {cache_path}")
        
    except Exception as e:
        print(f"[WARNING] Could not save output: {e}")

# Legacy compatibility function
def dynamic_module_output(lookup_id=None, output_path=None, module_key=None, provided_data=None):
    """
    Legacy compatibility wrapper for the old dynamic_module_output function
    
    This maintains backward compatibility while using the new DynamicHybridProcessor
    """
    # Extract text content from lookup_id if it's a file path
    text_content = ""
    
    if isinstance(provided_data, str):
        # If provided_data is string, use it directly
        text_content = provided_data
    elif lookup_id and os.path.exists(str(lookup_id)):
        # Try to read from file
        try:
            file_path = Path(lookup_id)
            ext = file_path.suffix.lower()
            
            if ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
            elif ext == '.docx':
                try:
                    import docx
                    doc = docx.Document(file_path)
                    text_content = '\n'.join([p.text for p in doc.paragraphs])
                except ImportError:
                    print("[WARNING] python-docx not available for DOCX reading")
                except Exception as e:
                    print(f"[WARNING] Could not read DOCX: {e}")
            elif ext == '.pdf':
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        text_content = '\n'.join(page.extract_text() or '' for page in reader.pages)
                except ImportError:
                    print("[WARNING] PyPDF2 not available for PDF reading")
                except Exception as e:
                    print(f"[WARNING] Could not read PDF: {e}")
        except Exception as e:
            print(f"[WARNING] Could not read file {lookup_id}: {e}")
    
    if not text_content:
        # If we still don't have content, check provided_data as dict
        if isinstance(provided_data, dict):
            # Try to find text content in provided_data
            text_content = provided_data.get('text', provided_data.get('content', ''))
    
    if not text_content:
        return {
            "status": "error",
            "error_message": "No text content available for processing",
            "success": False,
            "is_final": False
        }
    
    # Use the enhanced processing
    return enhanced_dynamic_module_output(
        text_content=text_content,
        module_key=module_key,
        output_path=output_path,
        metadata={"lookup_id": lookup_id, "provided_data": provided_data}
    )

# Function to test the new processor
def test_dynamic_hybrid_processor():
    """Test function for the new dynamic hybrid processor"""
    
    # Sample text for testing
    sample_text = """
    Interview with Sarah Johnson, age 32, conducted on June 8, 2025 at 123 Main Street.
    
    Sarah appeared clean and well-dressed during the interview. She was cooperative 
    and able to communicate clearly without any language barriers. No translation 
    services were required.
    
    Sarah stated that her household consists of herself, her husband Mark Johnson (age 34), 
    and their two children: Emma (age 8) and Jake (age 5). The family has lived at 
    123 Main Street for the past 3 years.
    
    Sarah works as a nurse at City Hospital and has been employed there for 5 years. 
    Mark works as a teacher at the local elementary school.
    
    Both parents denied any substance use or criminal history. They confirmed that 
    there are no firearms in the household and that all safety measures for the 
    children are in place.
    
    The children attend school regularly and are up to date with their medical care.
    """
    
    print("=== Testing Dynamic Hybrid Processor ===")
    
    # Test auto-detection
    processor = DynamicHybridProcessor()
    result = processor.process(sample_text)
    
    print(f"Auto-detected module type: {result['module']}")
    print(f"Confidence score: {result['confidence_score']}")
    print(f"Analysis method: {result['processing_approach']}")
    
    # Test specific module type
    result_adult = processor.process(sample_text, module_type="adult")
    print(f"\nForced adult module processing:")
    print(f"Module type: {result_adult['module']}")
    
    # Test legacy compatibility
    legacy_result = dynamic_module_output(
        provided_data=sample_text,
        module_key="adult"
    )
    
    print(f"\nLegacy compatibility test:")
    print(f"Status: {legacy_result['status']}")
    print(f"Narrative preview: {legacy_result['narrative'][:200]}...")
    
    return result, legacy_result

if __name__ == "__main__":
    test_dynamic_hybrid_processor()
