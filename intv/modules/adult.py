"""
Adult Module for INTV Pipeline - Hybrid Approach
Handles processing of adult-related documents and interactions using:
1. Python-based intelligent analysis for generic summaries
2. JSON-based policy constraints for structured output
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

class AdultModule:
    """Module for processing adult-related content using hybrid approach"""
    
    def __init__(self):
        self.module_name = "adult"
        self.description = "Processes adult-related content and documentation"
        self.variables_config = self._load_variables_config()
        
    def process(self, text_content, metadata=None):
        """
        HYBRID PROCESSING APPROACH:
        Phase 1: Python Analysis → Extract and analyze content intelligently
        Phase 2: JSON Variables → Structure output according to policy requirements
        
        Args:
            text_content (str): The extracted text content
            metadata (dict): Optional metadata about the document
            
        Returns:
            dict: Hybrid result with intelligent analysis + policy-structured output
        """
        # Phase 1: Intelligent Analysis (Python-based)
        generic_summary = self._create_generic_summary(text_content)
        
        # Phase 2: Policy-Structured Output (JSON-based)
        policy_output = self._apply_policy_structure(generic_summary, text_content)
        
        # Combined result
        result = {
            "module": self.module_name,
            "content_type": "adult",
            "original_length": len(text_content),
            "processing_approach": "hybrid",
            "generic_summary": generic_summary,
            "policy_structured_output": policy_output,
            "variables_config": self.variables_config
        }
        return result
    
    def _load_variables_config(self):
        """Load JSON variables configuration if available"""
        try:
            # Try to load from intv/modules first (main location)
            vars_path = Path(__file__).parent.parent / "intv" / "modules" / f"{self.module_name}_vars.json"
            if vars_path.exists():
                with open(vars_path, 'r') as f:
                    config = json.load(f)
                    # Convert old format to new format if needed
                    return self._convert_config_format(config)
            
            # Fallback to current directory
            vars_path = Path(__file__).parent / f"{self.module_name}_vars.json"
            if vars_path.exists():
                with open(vars_path, 'r') as f:
                    config = json.load(f)
                    return self._convert_config_format(config)
            
            return self._get_default_variables()
        except Exception as e:
            print(f"Warning: Could not load variables config: {e}")
            return self._get_default_variables()
    
    def _convert_config_format(self, config):
        """Convert old JSON format to new hybrid format"""
        # If already in new format, return as-is
        if "variables" in config:
            return config
        
        # Convert old format (flat key-value pairs) to new nested format
        converted = {
            "interview_type": "adult",
            "variables": {
                "personal_info": {},
                "family_structure": {},
                "assessment": {}
            }
        }
        
        # Map old keys to new structure
        key_mapping = {
            "Name": ("personal_info", "name"),
            "Age": ("personal_info", "age"),
            "Citizenship": ("personal_info", "citizenship"),
            "Role": ("family_structure", "relationship_status"),
            "Family": ("family_structure", "children"),
            "Relationship": ("family_structure", "family_influence"),
            "3010_Sign": ("assessment", "completed"),
            "Summary": ("assessment", "notes")
        }
        
        # Convert known keys
        for old_key, value_config in config.items():
            if old_key.startswith("_"):  # Skip metadata keys
                continue
                
            if old_key in key_mapping:
                category, new_key = key_mapping[old_key]
                converted["variables"][category][new_key] = {
                    "type": "text",
                    "default": value_config.get("default", ""),
                    "required": new_key in ["name", "completed"],
                    "hint": value_config.get("hint", "")
                }
        
        return converted
    
    def _get_default_variables(self):
        """Default variables configuration"""
        return {
            "interview_type": "adult",
            "variables": {
                "personal_info": {
                    "name": {"type": "text", "default": "", "required": True},
                    "age": {"type": "number", "default": "", "required": True},
                    "citizenship": {"type": "text", "default": "", "required": False}
                },
                "family_structure": {
                    "relationship_status": {"type": "text", "default": "", "required": False},
                    "children": {"type": "text", "default": "", "required": False},
                    "family_influence": {"type": "text", "default": "", "required": False}
                },
                "assessment": {
                    "completed": {"type": "boolean", "default": False, "required": True},
                    "notes": {"type": "text", "default": "", "required": False}
                }
            }
        }
    
    def _create_generic_summary(self, text_content):
        """
        Phase 1: Python-based intelligent analysis
        Extract structured information and create generic summary
        """
        # Extract personal information
        personal_info = self._extract_personal_info(text_content)
        
        # Extract family structure
        family_structure = self._extract_family_structure(text_content)
        
        # Extract assessment information
        assessment_info = self._extract_assessment_info(text_content)
        
        # Extract status indicators
        status_indicators = self._extract_status_indicators(text_content)
        
        # Create generic summary
        summary = {
            "personal_information": personal_info,
            "family_structure": family_structure,
            "assessment_information": assessment_info,
            "status_indicators": status_indicators,
            "document_analysis": {
                "content_length": len(text_content),
                "estimated_completeness": self._estimate_completeness(text_content),
                "key_sections_found": self._identify_key_sections(text_content)
            }
        }
        
        return summary
    
    def _apply_policy_structure(self, generic_summary, original_text):
        """
        Phase 2: JSON-based policy constraint application
        Structure the generic summary according to policy requirements
        """
        policy_output = {}
        
        # Apply variable structure from JSON config
        for category, variables in self.variables_config.get("variables", {}).items():
            policy_output[category] = {}
            
            for var_name, var_config in variables.items():
                # Extract value from generic summary or original text
                extracted_value = self._extract_value_for_variable(
                    var_name, var_config, generic_summary, original_text
                )
                
                policy_output[category][var_name] = {
                    "value": extracted_value,
                    "type": var_config.get("type", "text"),
                    "required": var_config.get("required", False),
                    "confidence": self._calculate_confidence(extracted_value, var_config)
                }
        
        return policy_output
    
    def _check_assessment_info(self, text):
        """Check for assessment-related information"""
        assessment_keywords = ["reviewed", "completed", "signed", "agreed", "assessment", "evaluation"]
        return any(keyword.lower() in text.lower() for keyword in assessment_keywords)
    
    def _check_personal_details(self, text):
        """Check for personal details"""
        personal_keywords = ["clean", "appropriate", "articulate", "language", "citizen", "resident"]
        return any(keyword.lower() in text.lower() for keyword in personal_keywords)
    
    def _check_relationship_info(self, text):
        """Check for relationship information"""
        relationship_keywords = ["girlfriend", "relationship", "kids", "children", "mom", "family"]
        return any(keyword.lower() in text.lower() for keyword in relationship_keywords)
    
    def _check_status_indicators(self, text):
        """Check for various status indicators"""
        status_keywords = ["military", "heritage", "influence", "barriers", "residence"]
        return any(keyword.lower() in text.lower() for keyword in status_keywords)
    
    def _analyze_structure(self, text):
        """Basic structure analysis"""
        lines = text.strip().split('\n')
        return {
            "line_count": len(lines),
            "has_multiple_items": len([line for line in lines if line.strip()]) > 1,
            "estimated_items": len([line for line in lines if line.strip()])
        }
    
    # ====== PHASE 1: INTELLIGENT DATA EXTRACTION METHODS ======
    
    def _extract_personal_info(self, text):
        """Extract personal information using intelligent pattern matching"""
        personal_info = {
            "name_mentioned": self._extract_names(text),
            "age_mentioned": self._extract_age(text),
            "citizenship_status": self._extract_citizenship(text),
            "language_proficiency": self._assess_language_quality(text),
            "personal_descriptors": self._extract_descriptors(text)
        }
        return personal_info
    
    def _extract_family_structure(self, text):
        """Extract family and relationship information"""
        family_info = {
            "relationship_mentions": self._extract_relationships(text),
            "children_mentioned": self._extract_children_info(text),
            "family_influence": self._assess_family_influence(text),
            "household_structure": self._assess_household_structure(text)
        }
        return family_info
    
    def _extract_assessment_info(self, text):
        """Extract assessment and evaluation information"""
        assessment_info = {
            "completion_indicators": self._find_completion_indicators(text),
            "assessment_notes": self._extract_assessment_notes(text),
            "evaluation_outcomes": self._extract_outcomes(text),
            "documentation_quality": self._assess_documentation_quality(text)
        }
        return assessment_info
    
    def _extract_status_indicators(self, text):
        """Extract various status and background indicators"""
        status_info = {
            "military_background": self._check_military_background(text),
            "cultural_heritage": self._extract_cultural_background(text),
            "residence_status": self._extract_residence_info(text),
            "barriers_mentioned": self._identify_barriers(text)
        }
        return status_info
    
    # ====== DETAILED EXTRACTION METHODS ======
    
    def _extract_names(self, text):
        """Extract potential names from text"""
        # Look for capitalized words that might be names
        name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        potential_names = re.findall(name_pattern, text)
        # Filter out common words that aren't names
        common_words = {'He', 'She', 'They', 'The', 'This', 'That', 'And', 'But', 'With', 'From'}
        names = [name for name in potential_names if name not in common_words]
        return names[:3]  # Return up to 3 potential names
    
    def _extract_age(self, text):
        """Extract age-related information"""
        age_patterns = [
            r'(\d{1,2})\s*years?\s*old',
            r'age\s*(?:of\s*)?(\d{1,2})',
            r'(\d{1,2})\s*y/?o'
        ]
        for pattern in age_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None
    
    def _extract_citizenship(self, text):
        """Extract citizenship or residency status"""
        citizenship_keywords = {
            'citizen': ['citizen', 'citizenship'],
            'resident': ['resident', 'residency', 'green card'],
            'immigrant': ['immigrant', 'immigration'],
            'heritage': ['heritage', 'background', 'origin']
        }
        
        found_status = []
        for status, keywords in citizenship_keywords.items():
            if any(keyword.lower() in text.lower() for keyword in keywords):
                found_status.append(status)
        return found_status
    
    def _assess_language_quality(self, text):
        """Assess language proficiency based on text quality"""
        quality_indicators = {
            'articulate': ['articulate', 'well-spoken', 'clear'],
            'appropriate': ['appropriate', 'professional', 'proper'],
            'clean': ['clean', 'neat', 'organized']
        }
        
        found_qualities = []
        for quality, keywords in quality_indicators.items():
            if any(keyword.lower() in text.lower() for keyword in keywords):
                found_qualities.append(quality)
        return found_qualities
    
    def _extract_relationships(self, text):
        """Extract relationship information"""
        relationship_patterns = {
            'romantic': ['girlfriend', 'boyfriend', 'partner', 'spouse', 'married'],
            'family': ['mother', 'father', 'mom', 'dad', 'sister', 'brother', 'family'],
            'children': ['kids', 'children', 'son', 'daughter', 'child']
        }
        
        found_relationships = {}
        for rel_type, keywords in relationship_patterns.items():
            matches = [keyword for keyword in keywords if keyword.lower() in text.lower()]
            if matches:
                found_relationships[rel_type] = matches
        return found_relationships
    
    def _extract_children_info(self, text):
        """Extract information about children"""
        child_indicators = []
        if any(word in text.lower() for word in ['kids', 'children', 'child']):
            child_indicators.append('children_mentioned')
        if any(word in text.lower() for word in ['son', 'daughter']):
            child_indicators.append('specific_children')
        return child_indicators
    
    def _find_completion_indicators(self, text):
        """Find indicators of completion or assessment"""
        completion_keywords = ['completed', 'finished', 'done', 'reviewed', 'assessed', 'evaluated']
        found_indicators = [keyword for keyword in completion_keywords 
                          if keyword.lower() in text.lower()]
        return found_indicators
    
    def _estimate_completeness(self, text):
        """Estimate how complete the document appears to be"""
        if len(text) < 100:
            return "minimal"
        elif len(text) < 500:
            return "partial"
        elif len(text) < 1000:
            return "substantial"
        else:
            return "comprehensive"
    
    def _identify_key_sections(self, text):
        """Identify key sections present in the document"""
        sections = []
        if any(word in text.lower() for word in ['name', 'personal', 'identity']):
            sections.append('personal_info')
        if any(word in text.lower() for word in ['family', 'relationship', 'children']):
            sections.append('family_structure')
        if any(word in text.lower() for word in ['assessment', 'evaluation', 'review']):
            sections.append('assessment')
        if any(word in text.lower() for word in ['military', 'heritage', 'background']):
            sections.append('background')
        return sections
    
    # ====== PHASE 2: POLICY VARIABLE EXTRACTION ======
    
    def _extract_value_for_variable(self, var_name, var_config, generic_summary, original_text):
        """Extract specific value for a policy variable from generic summary"""
        var_type = var_config.get("type", "text")
        
        # Map variable names to extraction methods
        extraction_map = {
            "name": lambda: self._get_name_from_summary(generic_summary),
            "age": lambda: self._get_age_from_summary(generic_summary),
            "citizenship": lambda: self._get_citizenship_from_summary(generic_summary),
            "relationship_status": lambda: self._get_relationship_from_summary(generic_summary),
            "children": lambda: self._get_children_from_summary(generic_summary),
            "family_influence": lambda: self._get_family_influence_from_summary(generic_summary),
            "completed": lambda: self._get_completion_from_summary(generic_summary),
            "notes": lambda: self._get_notes_from_summary(generic_summary, original_text)
        }
        
        if var_name in extraction_map:
            return extraction_map[var_name]()
        else:
            # Fallback: search original text for the variable name
            return self._search_text_for_variable(var_name, original_text, var_type)
    
    def _get_name_from_summary(self, summary):
        """Extract name from generic summary"""
        names = summary.get("personal_information", {}).get("name_mentioned", [])
        return names[0] if names else ""
    
    def _get_age_from_summary(self, summary):
        """Extract age from generic summary"""
        return summary.get("personal_information", {}).get("age_mentioned", "")
    
    def _get_citizenship_from_summary(self, summary):
        """Extract citizenship from generic summary"""
        citizenship = summary.get("personal_information", {}).get("citizenship_status", [])
        return ", ".join(citizenship) if citizenship else ""
    
    def _get_relationship_from_summary(self, summary):
        """Extract relationship status from generic summary"""
        relationships = summary.get("family_structure", {}).get("relationship_mentions", {})
        if "romantic" in relationships:
            return ", ".join(relationships["romantic"])
        return ""
    
    def _get_children_from_summary(self, summary):
        """Extract children information from generic summary"""
        children_info = summary.get("family_structure", {}).get("children_mentioned", [])
        return ", ".join(children_info) if children_info else ""
    
    def _get_family_influence_from_summary(self, summary):
        """Extract family influence from generic summary"""
        return summary.get("family_structure", {}).get("family_influence", "")
    
    def _get_completion_from_summary(self, summary):
        """Extract completion status from generic summary"""
        indicators = summary.get("assessment_information", {}).get("completion_indicators", [])
        return len(indicators) > 0
    
    def _get_notes_from_summary(self, summary, original_text):
        """Extract relevant notes from summary and original text"""
        notes = []
        
        # Add assessment notes
        assessment_notes = summary.get("assessment_information", {}).get("assessment_notes", [])
        notes.extend(assessment_notes)
        
        # Add document quality assessment
        quality = summary.get("document_analysis", {}).get("estimated_completeness", "")
        if quality:
            notes.append(f"Document completeness: {quality}")
        
        return "; ".join(notes) if notes else ""
    
    def _calculate_confidence(self, extracted_value, var_config):
        """Calculate confidence score for extracted value"""
        if not extracted_value:
            return 0.0
        
        if var_config.get("type") == "boolean":
            return 0.9 if extracted_value else 0.1
        
        if var_config.get("type") == "number":
            return 0.8 if isinstance(extracted_value, (int, float)) else 0.3
        
        # For text values, confidence based on length and content
        if isinstance(extracted_value, str):
            if len(extracted_value) > 20:
                return 0.8
            elif len(extracted_value) > 5:
                return 0.6
            else:
                return 0.4
        
        return 0.5  # Default confidence
    
    # ====== HELPER METHODS FOR ADDITIONAL EXTRACTION ======
    
    def _extract_descriptors(self, text):
        """Extract descriptive terms about the person"""
        descriptors = []
        descriptor_keywords = ['clean', 'appropriate', 'articulate', 'professional', 'polite']
        for keyword in descriptor_keywords:
            if keyword.lower() in text.lower():
                descriptors.append(keyword)
        return descriptors
    
    def _assess_family_influence(self, text):
        """Assess family influence mentioned in text"""
        influence_keywords = ['influence', 'support', 'help', 'guidance', 'pressure']
        found_influences = [keyword for keyword in influence_keywords 
                          if keyword.lower() in text.lower()]
        return ", ".join(found_influences) if found_influences else ""
    
    def _assess_household_structure(self, text):
        """Assess household structure from text"""
        if any(word in text.lower() for word in ['live with', 'lives with', 'household']):
            return "multi-person household"
        elif any(word in text.lower() for word in ['alone', 'by myself', 'independent']):
            return "independent living"
        return "unspecified"
    
    def _extract_assessment_notes(self, text):
        """Extract assessment-related notes"""
        notes = []
        if 'assessment' in text.lower():
            notes.append('Assessment mentioned')
        if any(word in text.lower() for word in ['reviewed', 'evaluated', 'assessed']):
            notes.append('Review process indicated')
        return notes
    
    def _extract_outcomes(self, text):
        """Extract evaluation outcomes"""
        outcomes = []
        outcome_keywords = ['approved', 'completed', 'passed', 'successful', 'satisfactory']
        for keyword in outcome_keywords:
            if keyword.lower() in text.lower():
                outcomes.append(keyword)
        return outcomes
    
    def _assess_documentation_quality(self, text):
        """Assess the quality of documentation"""
        if len(text) > 1000:
            return "comprehensive"
        elif len(text) > 500:
            return "adequate"
        elif len(text) > 100:
            return "basic"
        else:
            return "minimal"
    
    def _check_military_background(self, text):
        """Check for military background mentions"""
        military_keywords = ['military', 'army', 'navy', 'air force', 'marines', 'veteran', 'service']
        return any(keyword.lower() in text.lower() for keyword in military_keywords)
    
    def _extract_cultural_background(self, text):
        """Extract cultural or heritage information"""
        cultural_keywords = ['heritage', 'culture', 'background', 'origin', 'ethnicity']
        found_cultural = [keyword for keyword in cultural_keywords 
                         if keyword.lower() in text.lower()]
        return found_cultural
    
    def _extract_residence_info(self, text):
        """Extract residence and living situation information"""
        residence_info = []
        if any(word in text.lower() for word in ['resident', 'live', 'address', 'home']):
            residence_info.append('residence_mentioned')
        if any(word in text.lower() for word in ['citizen', 'citizenship']):
            residence_info.append('citizenship_status')
        return residence_info
    
    def _identify_barriers(self, text):
        """Identify potential barriers mentioned"""
        barrier_keywords = ['barrier', 'challenge', 'difficulty', 'problem', 'issue']
        found_barriers = [keyword for keyword in barrier_keywords 
                         if keyword.lower() in text.lower()]
        return found_barriers
    
    def _search_text_for_variable(self, var_name, text, var_type):
        """Fallback method to search text for variable content"""
        # Simple keyword search based on variable name
        if var_name.lower() in text.lower():
            # Try to extract context around the variable name
            import re
            pattern = rf'{re.escape(var_name.lower())}.{{0,50}}'
            match = re.search(pattern, text.lower())
            if match:
                return match.group(0)
        
        if var_type == "boolean":
            return False
        elif var_type == "number":
            return 0
        else:
            return ""

# Module interface for the pipeline
def get_module():
    """Return the module instance for pipeline integration"""
    return AdultModule()

# Metadata for module discovery
MODULE_INFO = {
    "name": "adult",
    "version": "2.0.0",
    "description": "Adult interview processing module with hybrid approach",
    "author": "INTV Pipeline",
    "supported_types": ["text", "pdf", "docx"],
    "requires_llm": False,
    "processing_approach": "hybrid",
    "features": [
        "intelligent_text_analysis",
        "policy_variable_extraction",
        "confidence_scoring",
        "generic_summary_generation"
    ]
}
