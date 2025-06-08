"""
Dynamic Hybrid Processor for INTV Pipeline
A universal processor that can handle any module configuration without requiring 
separate Python files for each module type.

This replaces the need for individual module.py files (adult.py, child.py, etc.)
with a single generalized processor that uses:
1. Universal extraction methods for intelligent analysis
2. JSON-based module configurations for policy structure
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

class DynamicHybridProcessor:
    """Universal hybrid processor for any module type"""
    
    def __init__(self, module_config_path: str = None):
        """
        Initialize the dynamic processor
        
        Args:
            module_config_path: Optional path to specific module config
        """
        self.available_modules = self._discover_modules()
        self.universal_extractors = self._initialize_universal_extractors()
        
    def process(self, text_content: str, module_type: str = None, metadata: Dict = None) -> Dict:
        """
        Universal hybrid processing for any module type
        
        Args:
            text_content: The text content to analyze
            module_type: The type of module (adult, child, casefile, etc.)
            metadata: Optional metadata about the document
            
        Returns:
            dict: Hybrid result with intelligent analysis + policy-structured output
        """
        # Auto-detect module type if not provided
        if not module_type:
            module_type = self._auto_detect_module_type(text_content)
        
        # Load module configuration
        module_config = self._load_module_config(module_type)
        
        # Phase 1: Universal Intelligent Analysis
        generic_analysis = self._create_universal_analysis(text_content, module_type)
        
        # Phase 2: Policy-Structured Output
        policy_output = self._apply_policy_structure(
            generic_analysis, text_content, module_config
        )
        
        # Combine results
        result = {
            "module": module_type,
            "processing_approach": "dynamic_hybrid",
            "content_length": len(text_content),
            "analysis_timestamp": datetime.now().isoformat(),
            "auto_detected": module_type if not module_type else False,
            "generic_analysis": generic_analysis,
            "policy_structured_output": policy_output,
            "module_config": module_config.get("_header", {}),
            "confidence_score": self._calculate_confidence(generic_analysis, policy_output)
        }
        
        return result
    
    def _discover_modules(self) -> List[str]:
        """Discover available module configurations"""
        modules_dir = Path(__file__).parent
        module_files = list(modules_dir.glob("*_vars.json"))
        
        modules = []
        for file_path in module_files:
            module_name = file_path.stem.replace("_vars", "")
            modules.append(module_name)
            
        return modules
    
    def _auto_detect_module_type(self, text_content: str) -> str:
        """Auto-detect the most appropriate module type based on content"""
        content_lower = text_content.lower()
        
        # Module detection patterns
        detection_patterns = {
            "adult": [
                "adult", "parent", "guardian", "mother", "father", "spouse",
                "employment", "job", "work", "household", "residence"
            ],
            "child": [
                "child", "minor", "school", "grade", "teacher", "playground",
                "daycare", "babysitter", "development", "pediatric"
            ],
            "casefile": [
                "case", "incident", "report", "investigation", "allegation",
                "cps", "dfps", "court", "legal", "evidence"
            ],
            "affidavit": [
                "affidavit", "sworn", "notary", "under oath", "penalty of perjury",
                "subscribed", "affirm", "declaration", "attestation"
            ],
            "collateral": [
                "witness", "third party", "collateral", "corroboration",
                "neighbor", "teacher", "doctor", "relative"
            ],
            "ar": [
                "alternative response", "family assessment", "group", "family unit",
                "voluntary", "services", "prevention", "support"
            ]
        }
        
        # Score each module type
        scores = {}
        for module_type, keywords in detection_patterns.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            scores[module_type] = score
        
        # Return the highest scoring module type, default to 'adult'
        best_match = max(scores.items(), key=lambda x: x[1])
        return best_match[0] if best_match[1] > 0 else "adult"
    
    def _load_module_config(self, module_type: str) -> Dict:
        """Load configuration for specified module type"""
        config_file = Path(__file__).parent / f"{module_type}_vars.json"
        
        if not config_file.exists():
            # Return default configuration if module-specific config doesn't exist
            return self._get_default_config(module_type)
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"Warning: Could not load config for {module_type}: {e}")
            return self._get_default_config(module_type)
    
    def _get_default_config(self, module_type: str) -> Dict:
        """Generate default configuration for unknown module types"""
        return {
            "_header": {
                "type": module_type,
                "label": f"{module_type.title()} Analysis",
                "description": f"Default configuration for {module_type} content analysis"
            },
            "Name": {"default": "[No Name]", "hint": "Name of primary subject"},
            "Date": {"default": "", "hint": "Date of document/interview"},
            "Location": {"default": "", "hint": "Location information"},
            "Summary": {"default": "", "hint": "Summary of content"},
            "Notes": {"default": "", "hint": "Additional notes and observations"}
        }
    
    def _initialize_universal_extractors(self) -> Dict:
        """Initialize universal extraction methods for all content types"""
        return {
            "personal_info": self._extract_personal_info,
            "dates_times": self._extract_dates_and_times,
            "locations": self._extract_locations,
            "relationships": self._extract_relationships,
            "demographics": self._extract_demographics,
            "contact_info": self._extract_contact_info,
            "legal_elements": self._extract_legal_elements,
            "safety_concerns": self._extract_safety_concerns,
            "behavioral_observations": self._extract_behavioral_observations,
            "family_structure": self._extract_family_structure,
            "employment_info": self._extract_employment_info,
            "education_info": self._extract_education_info,
            "medical_info": self._extract_medical_info,
            "substance_use": self._extract_substance_use,
            "criminal_history": self._extract_criminal_history,
            "services_history": self._extract_services_history,
            "financial_info": self._extract_financial_info,
            "housing_info": self._extract_housing_info,
            "documentation": self._extract_documentation_info,
            "narrative_flow": self._extract_narrative_flow
        }
    
    def _create_universal_analysis(self, text_content: str, module_type: str) -> Dict:
        """
        Phase 1: Universal intelligent analysis using all extraction methods
        This replaces the need for module-specific Python analysis
        """
        analysis = {
            "content_type": module_type,
            "analysis_method": "universal_extraction",
            "extracted_data": {}
        }
        
        # Run all universal extractors
        for extractor_name, extractor_func in self.universal_extractors.items():
            try:
                extracted_data = extractor_func(text_content)
                if extracted_data:  # Only include non-empty extractions
                    analysis["extracted_data"][extractor_name] = extracted_data
            except Exception as e:
                # Continue processing even if one extractor fails
                analysis["extracted_data"][extractor_name] = {
                    "error": str(e),
                    "status": "extraction_failed"
                }
        
        # Add content statistics
        analysis["content_statistics"] = {
            "word_count": len(text_content.split()),
            "character_count": len(text_content),
            "paragraph_count": len([p for p in text_content.split('\n\n') if p.strip()]),
            "sentence_count": len([s for s in re.split(r'[.!?]+', text_content) if s.strip()])
        }
        
        return analysis
    
    def _apply_policy_structure(self, analysis: Dict, text_content: str, module_config: Dict) -> Dict:
        """
        Phase 2: Apply policy structure based on module configuration
        Maps universal analysis to module-specific output format
        """
        policy_output = {}
        
        # Process each variable in the module configuration
        for var_name, var_config in module_config.items():
            if var_name.startswith("_"):  # Skip metadata
                continue
                
            # Try to populate variable from analysis
            value = self._map_analysis_to_variable(
                var_name, var_config, analysis, text_content
            )
            
            policy_output[var_name] = {
                "value": value,
                "default": var_config.get("default", ""),
                "hint": var_config.get("hint", ""),
                "confidence": self._calculate_variable_confidence(value, var_config),
                "source": self._identify_data_source(var_name, analysis)
            }
        
        return policy_output
    
    def _map_analysis_to_variable(self, var_name: str, var_config: Dict, 
                                  analysis: Dict, text_content: str) -> str:
        """Map extracted analysis data to specific policy variables"""
        extracted_data = analysis.get("extracted_data", {})
        
        # Variable name mapping strategies
        var_lower = var_name.lower()
        
        # Direct mapping attempts
        if "name" in var_lower:
            names = extracted_data.get("personal_info", {}).get("names", [])
            return names[0] if names else var_config.get("default", "")
        
        elif "date" in var_lower:
            dates = extracted_data.get("dates_times", {}).get("dates", [])
            return dates[0] if dates else var_config.get("default", "")
        
        elif "location" in var_lower or "address" in var_lower:
            locations = extracted_data.get("locations", {}).get("addresses", [])
            return locations[0] if locations else var_config.get("default", "")
        
        elif "family" in var_lower or "household" in var_lower:
            family_info = extracted_data.get("family_structure", {})
            return family_info.get("summary", var_config.get("default", ""))
        
        elif "employment" in var_lower or "job" in var_lower:
            employment = extracted_data.get("employment_info", {})
            return employment.get("current_job", var_config.get("default", ""))
        
        elif "behavior" in var_lower or "appearance" in var_lower:
            behavioral = extracted_data.get("behavioral_observations", {})
            return behavioral.get("summary", var_config.get("default", ""))
        
        elif "communication" in var_lower:
            behavioral = extracted_data.get("behavioral_observations", {})
            return behavioral.get("communication_style", var_config.get("default", ""))
        
        # If no specific mapping found, try keyword search in text
        hint = var_config.get("hint", "").lower()
        if hint:
            # Look for content related to the hint
            relevant_content = self._extract_content_by_keywords(text_content, hint)
            return relevant_content if relevant_content else var_config.get("default", "")
        
        return var_config.get("default", "")
    
    def _extract_content_by_keywords(self, text_content: str, hint: str) -> str:
        """Extract content based on hint keywords"""
        # Extract key words from hint
        keywords = [word.strip() for word in hint.split() 
                   if len(word) > 3 and word not in ['the', 'and', 'for', 'with']]
        
        if not keywords:
            return ""
        
        # Find sentences containing keywords
        sentences = [s.strip() for s in re.split(r'[.!?]+', text_content) if s.strip()]
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in keywords):
                relevant_sentences.append(sentence)
        
        return ". ".join(relevant_sentences[:2])  # Return first 2 relevant sentences
    
    # Universal extraction methods start here
    def _extract_personal_info(self, text: str) -> Dict:
        """Extract personal information using universal patterns"""
        info = {"names": [], "ages": [], "genders": [], "identifiers": []}
        
        # Name patterns
        name_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)\s+(?:is|was|stated|said|reported)',
            r'(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
            r'\b([A-Z][a-z]+)\s+(?:aged|age)\s+\d+',
            r'Interview(?:ee|ed):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)'
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, text)
            info["names"].extend(matches)
        
        # Age patterns
        age_patterns = [
            r'(?:aged|age)\s+(\d+)',
            r'(\d+)\s*(?:years?\s*old|y\.?o\.?)',
            r'\b(\d{1,2})\s*(?:year|yr)',
        ]
        
        for pattern in age_patterns:
            matches = re.findall(pattern, text)
            info["ages"].extend([int(age) for age in matches if 0 <= int(age) <= 120])
        
        return info
    
    def _extract_dates_and_times(self, text: str) -> Dict:
        """Extract dates and times"""
        dates_times = {"dates": [], "times": []}
        
        # Date patterns
        date_patterns = [
            r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b',
            r'\b([A-Za-z]+\s+\d{1,2},?\s+\d{4})\b',
            r'\b(\d{1,2}\s+[A-Za-z]+\s+\d{4})\b'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            dates_times["dates"].extend(matches)
        
        # Time patterns
        time_patterns = [
            r'\b(\d{1,2}:\d{2}(?:\s*[AP]M)?)\b',
            r'\b(\d{1,2}\s*[AP]M)\b'
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates_times["times"].extend(matches)
        
        return dates_times
    
    def _extract_locations(self, text: str) -> Dict:
        """Extract location information"""
        locations = {"addresses": [], "cities": [], "states": [], "facilities": []}
        
        # Address patterns
        address_patterns = [
            r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\.?\b',
            r'\b\d+\s+[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}\b'
        ]
        
        for pattern in address_patterns:
            matches = re.findall(pattern, text)
            locations["addresses"].extend(matches)
        
        # State abbreviations
        state_pattern = r'\b([A-Z]{2})\s+\d{5}\b'
        state_matches = re.findall(state_pattern, text)
        locations["states"].extend(state_matches)
        
        return locations
    
    def _extract_relationships(self, text: str) -> Dict:
        """Extract relationship information"""
        relationships = {"family_relations": [], "professional_relations": []}
        
        # Family relationship patterns
        family_terms = ["mother", "father", "parent", "son", "daughter", "child", 
                       "husband", "wife", "spouse", "brother", "sister", "sibling",
                       "uncle", "aunt", "cousin", "grandmother", "grandfather", "stepfather", "stepmother"]
        
        for term in family_terms:
            pattern = rf'\b(?:his|her|their|the)\s+{term}\b'
            if re.search(pattern, text, re.IGNORECASE):
                relationships["family_relations"].append(term)
        
        return relationships
    
    def _extract_demographics(self, text: str) -> Dict:
        """Extract demographic information"""
        demographics = {"ethnicity": [], "citizenship": [], "language": []}
        
        # Citizenship patterns
        citizenship_patterns = [
            r'\b(US citizen|American citizen|naturalized citizen)\b',
            r'\b(citizen of [A-Za-z]+)\b'
        ]
        
        for pattern in citizenship_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            demographics["citizenship"].extend(matches)
        
        return demographics
    
    def _extract_contact_info(self, text: str) -> Dict:
        """Extract contact information"""
        contact = {"phones": [], "emails": []}
        
        # Phone patterns
        phone_patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            r'\(\d{3}\)\s*\d{3}[-.]?\d{4}\b'
        ]
        
        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            contact["phones"].extend(matches)
        
        # Email patterns
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_matches = re.findall(email_pattern, text)
        contact["emails"].extend(email_matches)
        
        return contact
    
    def _extract_legal_elements(self, text: str) -> Dict:
        """Extract legal elements and references"""
        legal = {"statutes": [], "forms": [], "procedures": []}
        
        # Legal form patterns
        form_patterns = [
            r'\b(Form\s+\d+)\b',
            r'\b(DFPS\s+Form\s+\d+)\b',
            r'\b(\d{4}\s+form)\b'
        ]
        
        for pattern in form_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            legal["forms"].extend(matches)
        
        return legal
    
    def _extract_safety_concerns(self, text: str) -> Dict:
        """Extract safety-related information"""
        safety = {"concerns": [], "protective_factors": []}
        
        safety_keywords = ["danger", "risk", "unsafe", "threat", "harm", "violence", 
                          "abuse", "neglect", "injury", "weapon", "gun", "firearm"]
        
        for keyword in safety_keywords:
            if keyword in text.lower():
                # Find sentences containing safety keywords
                sentences = re.split(r'[.!?]+', text)
                for sentence in sentences:
                    if keyword in sentence.lower():
                        safety["concerns"].append(sentence.strip())
                        break
        
        return safety
    
    def _extract_behavioral_observations(self, text: str) -> Dict:
        """Extract behavioral observations"""
        behavioral = {"appearance": "", "behavior": "", "communication_style": "", "summary": ""}
        
        # Look for appearance descriptions
        appearance_keywords = ["appeared", "appearance", "looked", "dressed", "clean", "unkempt"]
        behavior_keywords = ["behavior", "acted", "seemed", "cooperative", "hostile"]
        communication_keywords = ["spoke", "communication", "language", "articulated", "expressed"]
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in appearance_keywords):
                behavioral["appearance"] = sentence
            elif any(keyword in sentence_lower for keyword in behavior_keywords):
                behavioral["behavior"] = sentence
            elif any(keyword in sentence_lower for keyword in communication_keywords):
                behavioral["communication_style"] = sentence
        
        # Create summary
        observations = [v for v in behavioral.values() if v]
        behavioral["summary"] = " ".join(observations)
        
        return behavioral
    
    def _extract_family_structure(self, text: str) -> Dict:
        """Extract family structure information"""
        family = {"household_members": [], "relationships": [], "summary": ""}
        
        # Look for household composition
        household_patterns = [
            r'household consists of ([^.!?]+)',
            r'family includes ([^.!?]+)',
            r'living with ([^.!?]+)'
        ]
        
        for pattern in household_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            family["household_members"].extend(matches)
        
        if family["household_members"]:
            family["summary"] = f"Household composition: {', '.join(family['household_members'])}"
        
        return family
    
    def _extract_employment_info(self, text: str) -> Dict:
        """Extract employment information"""
        employment = {"current_job": "", "employer": "", "work_history": []}
        
        # Employment patterns
        job_patterns = [
            r'(?:works as|employed as|job as)\s+([^.!?]+)',
            r'(?:employed at|works at)\s+([^.!?]+)',
            r'occupation:\s*([^.!?]+)'
        ]
        
        for pattern in job_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                employment["current_job"] = matches[0].strip()
                break
        
        return employment
    
    def _extract_education_info(self, text: str) -> Dict:
        """Extract education information"""
        education = {"level": "", "school": "", "grade": ""}
        
        education_patterns = [
            r'(?:grade|school):\s*([^.!?]+)',
            r'(?:attends|enrolled at)\s+([^.!?]+)',
            r'(?:high school|college|university|elementary)'
        ]
        
        for pattern in education_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                education["school"] = matches[0].strip()
                break
        
        return education
    
    def _extract_medical_info(self, text: str) -> Dict:
        """Extract medical information"""
        medical = {"conditions": [], "medications": [], "providers": []}
        
        medical_keywords = ["doctor", "medication", "prescription", "diagnosis", 
                           "treatment", "therapy", "medical", "health"]
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in medical_keywords):
                medical["conditions"].append(sentence)
        
        return medical
    
    def _extract_substance_use(self, text: str) -> Dict:
        """Extract substance use information"""
        substance = {"alcohol": [], "drugs": [], "tobacco": []}
        
        substance_keywords = {
            "alcohol": ["alcohol", "drinking", "beer", "wine", "liquor"],
            "drugs": ["drugs", "marijuana", "cocaine", "heroin", "methamphetamine"],
            "tobacco": ["tobacco", "smoking", "cigarettes", "vaping"]
        }
        
        for category, keywords in substance_keywords.items():
            for keyword in keywords:
                if keyword in text.lower():
                    substance[category].append(f"Reference to {keyword} found")
        
        return substance
    
    def _extract_criminal_history(self, text: str) -> Dict:
        """Extract criminal history information"""
        criminal = {"arrests": [], "convictions": [], "pending_cases": []}
        
        criminal_keywords = ["arrest", "conviction", "charge", "court", "probation", 
                           "parole", "sentence", "jail", "prison"]
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in criminal_keywords):
                criminal["arrests"].append(sentence)
        
        return criminal
    
    def _extract_services_history(self, text: str) -> Dict:
        """Extract services history"""
        services = {"current_services": [], "past_services": [], "referrals": []}
        
        service_keywords = ["services", "therapy", "counseling", "support", "assistance", 
                           "program", "intervention", "treatment"]
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in service_keywords):
                services["current_services"].append(sentence)
        
        return services
    
    def _extract_financial_info(self, text: str) -> Dict:
        """Extract financial information"""
        financial = {"income": [], "assistance": [], "employment_status": ""}
        
        financial_keywords = ["income", "salary", "wages", "benefits", "assistance", 
                             "welfare", "unemployment", "disability"]
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in financial_keywords):
                financial["income"].append(sentence)
        
        return financial
    
    def _extract_housing_info(self, text: str) -> Dict:
        """Extract housing information"""
        housing = {"type": "", "condition": "", "safety": []}
        
        housing_keywords = ["house", "apartment", "residence", "home", "living", 
                           "housing", "rent", "mortgage"]
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in housing_keywords):
                housing["condition"] = sentence
                break
        
        return housing
    
    def _extract_documentation_info(self, text: str) -> Dict:
        """Extract documentation references"""
        documentation = {"forms_completed": [], "signatures": [], "consents": []}
        
        doc_keywords = ["form", "document", "signature", "signed", "consent", 
                       "agreement", "acknowledgment"]
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in doc_keywords):
                documentation["forms_completed"].append(sentence)
        
        return documentation
    
    def _extract_narrative_flow(self, text: str) -> Dict:
        """Extract narrative structure and flow"""
        narrative = {
            "introduction": "",
            "main_content": "",
            "conclusion": "",
            "key_points": []
        }
        
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if paragraphs:
            narrative["introduction"] = paragraphs[0]
            if len(paragraphs) > 2:
                narrative["main_content"] = '\n\n'.join(paragraphs[1:-1])
                narrative["conclusion"] = paragraphs[-1]
            elif len(paragraphs) == 2:
                narrative["conclusion"] = paragraphs[1]
        
        return narrative
    
    def _calculate_confidence(self, analysis: Dict, policy_output: Dict) -> float:
        """Calculate overall confidence score for the analysis"""
        extracted_count = len([v for v in analysis.get("extracted_data", {}).values() 
                              if v and not isinstance(v, dict) or not v.get("error")])
        total_extractors = len(self.universal_extractors)
        
        populated_vars = len([v for v in policy_output.values() 
                             if v.get("value") and v["value"] != v.get("default", "")])
        total_vars = len(policy_output)
        
        extraction_score = extracted_count / total_extractors if total_extractors > 0 else 0
        population_score = populated_vars / total_vars if total_vars > 0 else 0
        
        return round((extraction_score + population_score) / 2, 3)
    
    def _calculate_variable_confidence(self, value: str, var_config: Dict) -> float:
        """Calculate confidence score for individual variable"""
        if not value or value == var_config.get("default", ""):
            return 0.0
        
        # Higher confidence for longer, more specific values
        if len(value) > 50:
            return 0.9
        elif len(value) > 20:
            return 0.7
        elif len(value) > 5:
            return 0.5
        else:
            return 0.3
    
    def _identify_data_source(self, var_name: str, analysis: Dict) -> str:
        """Identify which extraction method provided the data for a variable"""
        extracted_data = analysis.get("extracted_data", {})
        
        var_lower = var_name.lower()
        
        if "name" in var_lower and "personal_info" in extracted_data:
            return "personal_info"
        elif "date" in var_lower and "dates_times" in extracted_data:
            return "dates_times"
        elif "location" in var_lower and "locations" in extracted_data:
            return "locations"
        elif "family" in var_lower and "family_structure" in extracted_data:
            return "family_structure"
        elif "behavior" in var_lower and "behavioral_observations" in extracted_data:
            return "behavioral_observations"
        else:
            return "universal_extraction"

    def get_available_modules(self) -> List[str]:
        """Get list of available module types"""
        return self.available_modules
    
    def validate_module_config(self, module_type: str) -> bool:
        """Validate that a module configuration exists and is valid"""
        try:
            config = self._load_module_config(module_type)
            return "_header" in config or len(config) > 0
        except:
            return False
