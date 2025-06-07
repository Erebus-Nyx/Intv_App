"""
Affidavit Module for INTV Pipeline - Hybrid Approach
Combines Python-based intelligent analysis with JSON-based policy constraints
"""

import json
import os
import re
from pathlib import Path

class AffidavitModule:
    """Module for processing affidavit-related content using hybrid approach"""
    
    def __init__(self):
        self.module_name = "affidavit"
        self.description = "Processes affidavit documents using hybrid Python + JSON approach"
        self.variables_config = self._load_variables_config()
    
    def _load_variables_config(self):
        """Load and process the JSON variables configuration"""
        # Look for config in intv/modules directory
        config_paths = [
            Path(__file__).parent.parent / "intv" / "modules" / "affidavit_vars.json",
            Path(__file__).parent / "affidavit_vars.json"
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Remove comments from JSON
                        lines = content.split('\n')
                        cleaned_lines = [line for line in lines if not line.strip().startswith('//')]
                        cleaned_content = '\n'.join(cleaned_lines)
                        config = json.loads(cleaned_content)
                        
                        # Convert old format to hybrid format if needed
                        return self._convert_config_format(config)
                except Exception as e:
                    print(f"Warning: Could not load config from {config_path}: {e}")
        
        # Return minimal default config if no file found
        return self._get_default_config()
    
    def _convert_config_format(self, old_config):
        """Convert old JSON format to new hybrid format"""
        if '_hybrid_config' in old_config:
            return old_config
            
        # Create hybrid config with intelligent defaults
        hybrid_config = old_config.copy()
        hybrid_config['_hybrid_config'] = {
            'confidence_threshold': 0.75,
            'extraction_strategies': {
                'affiant_info': ['name_recognition', 'title_extraction', 'contact_analysis'],
                'legal_elements': ['oath_detection', 'signature_analysis', 'notary_identification'],
                'content_analysis': ['fact_extraction', 'purpose_identification', 'subject_classification'],
                'formal_elements': ['date_extraction', 'jurisdiction_analysis', 'legal_formatting']
            },
            'phase_weights': {
                'intelligent_analysis': 0.65,
                'policy_structure': 0.35
            }
        }
        return hybrid_config
    
    def _get_default_config(self):
        """Return default configuration if no file found"""
        return {
            '_header': {
                'type': 'affidavit',
                'label': 'Affidavit Document',
                'description': 'Default configuration for affidavit processing'
            },
            'DocumentType': {'default': 'Affidavit', 'hint': 'Type of legal document'},
            'Affiant': {'default': '', 'hint': 'Name of person making the affidavit'},
            'Subject': {'default': '', 'hint': 'Subject matter of the affidavit'},
            'Summary': {'default': 'Affidavit document processed', 'hint': 'Summary of affidavit contents'},
            '_hybrid_config': {
                'confidence_threshold': 0.75,
                'extraction_strategies': {
                    'affiant_info': ['name_recognition'],
                    'legal_elements': ['oath_detection'],
                    'content_analysis': ['fact_extraction']
                }
            }
        }
        
    def process(self, text_content, metadata=None):
        """
        Process affidavit content using hybrid approach
        
        Phase 1: Python-based intelligent analysis
        Phase 2: JSON-based policy constraint application
        
        Returns:
            dict: Results with both generic summary and policy-structured output
        """
        # PHASE 1: Python-based intelligent analysis
        generic_summary = self._create_generic_summary(text_content, metadata)
        
        # PHASE 2: JSON-based policy structure application
        policy_structured_output = self._apply_policy_structure(text_content, generic_summary, metadata)
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(generic_summary, policy_structured_output)
        
        return {
            "module": self.module_name,
            "approach": "hybrid_python_json",
            "content_type": "affidavit",
            "processing_phases": {
                "phase_1": "python_intelligent_analysis", 
                "phase_2": "json_policy_constraints"
            },
            "generic_summary": generic_summary,
            "policy_structured_output": policy_structured_output,
            "confidence_score": confidence,
            "metadata": metadata or {}
        }
    
    def _check_signature_elements(self, text):
        """Check for common signature/verification elements"""
        signature_keywords = ["signed", "sworn", "notarized", "witnessed", "signature"]
        return any(keyword.lower() in text.lower() for keyword in signature_keywords)
    
    def _check_oath_language(self, text):
        """Check for oath or affirmation language"""
        oath_keywords = ["swear", "affirm", "under penalty", "perjury", "true and correct"]
        return any(keyword.lower() in text.lower() for keyword in oath_keywords)
    
    def _analyze_structure(self, text):
        """Basic structure analysis"""
        lines = text.strip().split('\n')
        return {
            "line_count": len(lines),
            "has_multiple_paragraphs": len([line for line in lines if line.strip()]) > 1,
            "estimated_sections": len([line for line in lines if line.strip() and not line.startswith(' ')])
        }
    
    def _create_generic_summary(self, text_content, metadata=None):
        """
        Phase 1: Python-based intelligent analysis
        Extract meaningful information using intelligent Python processing
        """
        summary = {
            "intelligent_analysis": {
                "affiant_information": self._extract_affiant_info(text_content),
                "legal_elements": self._extract_legal_elements(text_content),
                "content_analysis": self._analyze_content(text_content),
                "formal_elements": self._extract_formal_elements(text_content),
                "document_structure": self._analyze_document_structure(text_content),
                "verification_elements": self._extract_verification_elements(text_content)
            },
            "extraction_metadata": {
                "text_length": len(text_content),
                "processing_strategies": list(self.variables_config.get('_hybrid_config', {}).get('extraction_strategies', {}).keys()),
                "confidence_factors": self._assess_confidence_factors(text_content)
            }
        }
        return summary
    
    def _apply_policy_structure(self, text_content, generic_summary, metadata=None):
        """
        Phase 2: JSON-based policy constraint application
        Structure the output according to policy requirements using JSON variables
        """
        # Extract variable definitions (excluding metadata)
        variables = {k: v for k, v in self.variables_config.items() 
                    if not k.startswith('_')}
        
        # Map intelligent analysis to policy variables
        structured_output = {}
        for var_name, var_config in variables.items():
            extracted_value = self._map_to_policy_variable(
                var_name, generic_summary, text_content
            )
            
            structured_output[var_name] = {
                "value": extracted_value or var_config.get("default", ""),
                "confidence": self._calculate_field_confidence(var_name, extracted_value, text_content),
                "source": "hybrid_extraction",
                "hint": var_config.get("hint", "")
            }
        
        return structured_output
    
    def _extract_affiant_info(self, text):
        """Extract information about the affiant (person making the affidavit)"""
        affiant_info = {
            "names": self._extract_affiant_names(text),
            "titles": self._extract_titles(text),
            "contact_info": self._extract_contact_info(text),
            "credentials": self._extract_credentials(text)
        }
        return affiant_info
    
    def _extract_legal_elements(self, text):
        """Extract legal elements of the affidavit"""
        legal_elements = {
            "oath_language": self._detect_oath_language(text),
            "signature_elements": self._detect_signature_elements(text),
            "notary_elements": self._detect_notary_elements(text),
            "legal_basis": self._extract_legal_basis(text),
            "jurisdiction": self._extract_jurisdiction(text)
        }
        return legal_elements
    
    def _analyze_content(self, text):
        """Analyze the substantive content of the affidavit"""
        content_analysis = {
            "subject_matter": self._identify_subject_matter(text),
            "factual_statements": self._extract_factual_statements(text),
            "purpose": self._identify_purpose(text),
            "key_facts": self._extract_key_facts(text),
            "supporting_documents": self._identify_attachments(text)
        }
        return content_analysis
    
    def _extract_formal_elements(self, text):
        """Extract formal legal document elements"""
        formal_elements = {
            "document_date": self._extract_document_date(text),
            "sworn_date": self._extract_sworn_date(text),
            "venue": self._extract_venue(text),
            "case_references": self._extract_case_references(text),
            "legal_captions": self._extract_legal_captions(text)
        }
        return formal_elements
    
    def _extract_verification_elements(self, text):
        """Extract verification and authentication elements"""
        verification = {
            "notary_info": self._extract_notary_info(text),
            "witness_info": self._extract_witness_info(text),
            "seal_references": self._detect_seal_references(text),
            "commission_info": self._extract_commission_info(text)
        }
        return verification
    
    def _extract_affiant_names(self, text):
        """Extract the name(s) of the affiant(s)"""
        names = []
        
        # Look for "I, [Name]" patterns common in affidavits
        i_patterns = [
            r'I,\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'I am\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'My name is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        ]
        
        for pattern in i_patterns:
            matches = re.findall(pattern, text)
            names.extend(matches)
        
        # Look for signature lines
        signature_patterns = [
            r'_+\s*\n\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'Signed:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'Affiant:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        ]
        
        for pattern in signature_patterns:
            matches = re.findall(pattern, text)
            names.extend(matches)
        
        return list(set(names))
    
    def _extract_titles(self, text):
        """Extract titles or roles of the affiant"""
        titles = []
        title_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),?\s+(?:duly sworn|being sworn)',
            r'I am (?:a|an|the)\s+([a-z]+(?:\s+[a-z]+)*)',
            r'in my capacity as\s+([a-z]+(?:\s+[a-z]+)*)'
        ]
        
        for pattern in title_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            titles.extend(matches)
        
        return list(set(titles))
    
    def _extract_contact_info(self, text):
        """Extract contact information"""
        contact_info = []
        
        # Phone numbers
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        phones = re.findall(phone_pattern, text)
        contact_info.extend([f"Phone: {phone}" for phone in phones])
        
        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        contact_info.extend([f"Email: {email}" for email in emails])
        
        # Addresses (basic pattern)
        address_pattern = r'\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr)'
        addresses = re.findall(address_pattern, text)
        contact_info.extend([f"Address: {addr}" for addr in addresses])
        
        return contact_info
    
    def _extract_credentials(self, text):
        """Extract professional credentials"""
        credentials = []
        credential_patterns = [
            r'\b([A-Z]{2,4})\b(?:\s+#?\s*\d+)?',  # License abbreviations
            r'License\s+(?:No\.?\s*)?(\d+)',
            r'Certificate\s+(?:No\.?\s*)?(\d+)'
        ]
        
        for pattern in credential_patterns:
            matches = re.findall(pattern, text)
            credentials.extend(matches)
        
        return list(set(credentials))
    
    def _detect_oath_language(self, text):
        """Detect oath or affirmation language"""
        oath_phrases = [
            'solemnly swear', 'solemnly affirm', 'under penalty of perjury',
            'true and correct', 'swear or affirm', 'oath or affirmation'
        ]
        
        detected_oaths = []
        for phrase in oath_phrases:
            if phrase.lower() in text.lower():
                detected_oaths.append(phrase.title())
        
        return detected_oaths
    
    def _detect_signature_elements(self, text):
        """Detect signature-related elements"""
        signature_elements = []
        signature_indicators = [
            'signature', 'signed', 'executed', 'subscribed', 'sworn before me'
        ]
        
        for indicator in signature_indicators:
            if indicator.lower() in text.lower():
                signature_elements.append(indicator.title())
        
        return signature_elements
    
    def _detect_notary_elements(self, text):
        """Detect notary public elements"""
        notary_elements = []
        notary_indicators = [
            'notary public', 'notarized', 'acknowledged', 'sworn before me',
            'my commission expires', 'notarial seal', 'official seal'
        ]
        
        for indicator in notary_indicators:
            if indicator.lower() in text.lower():
                notary_elements.append(indicator.title())
        
        return notary_elements
    
    def _extract_legal_basis(self, text):
        """Extract legal basis or authority for the affidavit"""
        legal_basis = []
        basis_patterns = [
            r'pursuant to\s+([^.]+)',
            r'under\s+(?:the\s+)?([A-Z][^.]+)',
            r'in accordance with\s+([^.]+)'
        ]
        
        for pattern in basis_patterns:
            matches = re.findall(pattern, text)
            legal_basis.extend(matches)
        
        return legal_basis
    
    def _extract_jurisdiction(self, text):
        """Extract jurisdiction information"""
        jurisdictions = []
        
        # State patterns
        state_pattern = r'State of\s+([A-Z][a-z]+)'
        states = re.findall(state_pattern, text)
        jurisdictions.extend([f"State of {state}" for state in states])
        
        # County patterns
        county_pattern = r'County of\s+([A-Z][a-z]+)'
        counties = re.findall(county_pattern, text)
        jurisdictions.extend([f"County of {county}" for county in counties])
        
        return jurisdictions

    def _identify_subject_matter(self, text):
        """Identify the subject matter of the affidavit"""
        subjects = []
        
        # Look for common affidavit subjects
        subject_indicators = {
            'Identity': ['identity', 'identification', 'who I am'],
            'Residency': ['residence', 'domicile', 'live at', 'reside'],
            'Employment': ['employment', 'work', 'employed by', 'job'],
            'Financial': ['income', 'assets', 'financial', 'earnings'],
            'Family': ['marriage', 'divorce', 'children', 'family'],
            'Legal Proceedings': ['lawsuit', 'case', 'court', 'legal action'],
            'Business': ['business', 'company', 'corporation', 'partnership'],
            'Property': ['property', 'real estate', 'ownership', 'title']
        }
        
        for subject, keywords in subject_indicators.items():
            if any(keyword.lower() in text.lower() for keyword in keywords):
                subjects.append(subject)
        
        return subjects
    
    def _extract_factual_statements(self, text):
        """Extract key factual statements"""
        factual_statements = []
        
        # Look for numbered statements
        numbered_pattern = r'(\d+)\.\s*([^.]+\.)'
        numbered_facts = re.findall(numbered_pattern, text)
        factual_statements.extend([f"{num}. {fact}" for num, fact in numbered_facts])
        
        # Look for "I state that" or similar patterns
        statement_patterns = [
            r'I state that\s+([^.]+\.)',
            r'I declare that\s+([^.]+\.)',
            r'I certify that\s+([^.]+\.)'
        ]
        
        for pattern in statement_patterns:
            matches = re.findall(pattern, text)
            factual_statements.extend(matches)
        
        return factual_statements
    
    def _identify_purpose(self, text):
        """Identify the purpose of the affidavit"""
        purposes = []
        purpose_patterns = [
            r'for the purpose of\s+([^.]+)',
            r'this affidavit is made\s+([^.]+)',
            r'in support of\s+([^.]+)'
        ]
        
        for pattern in purpose_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            purposes.extend(matches)
        
        return purposes
    
    def _extract_key_facts(self, text):
        """Extract key facts from the affidavit"""
        key_facts = []
        
        # Look for important fact indicators
        fact_indicators = ['the fact is', 'it is true that', 'I know that', 'I have personal knowledge']
        
        for indicator in fact_indicators:
            if indicator.lower() in text.lower():
                # Extract text following the indicator
                pattern = rf'{re.escape(indicator.lower())}\s+([^.]+\.)'
                matches = re.findall(pattern, text, re.IGNORECASE)
                key_facts.extend(matches)
        
        return key_facts
    
    def _identify_attachments(self, text):
        """Identify attachments or exhibits referenced"""
        attachments = []
        attachment_patterns = [
            r'Exhibit\s+([A-Z])',
            r'Attachment\s+(\d+)',
            r'attached hereto as\s+([^.]+)',
            r'see attached\s+([^.]+)'
        ]
        
        for pattern in attachment_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            attachments.extend(matches)
        
        return attachments
    
    def _extract_document_date(self, text):
        """Extract the document creation date"""
        dates = self._extract_all_dates(text)
        
        # Look for specific date patterns near "dated" or similar
        date_context_patterns = [
            r'dated\s+([^.]+)',
            r'this\s+\d+\w*\s+day of\s+([^.]+)',
            r'executed on\s+([^.]+)'
        ]
        
        for pattern in date_context_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0]
        
        # Return first date found if no specific context
        return dates[0] if dates else ""
    
    def _extract_sworn_date(self, text):
        """Extract the date the affidavit was sworn"""
        sworn_patterns = [
            r'sworn (?:to )?(?:before me )?(?:this )?\s*([^.]+)',
            r'subscribed and sworn\s+([^.]+)',
            r'acknowledged before me\s+([^.]+)'
        ]
        
        for pattern in sworn_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0]
        
        return ""
    
    def _extract_venue(self, text):
        """Extract venue information"""
        venues = []
        
        # Look for formal venue statements
        venue_patterns = [
            r'STATE OF\s+([A-Z]+)\s*COUNTY OF\s+([A-Z]+)',
            r'(?:State|County) of\s+([A-Z][a-z]+)'
        ]
        
        for pattern in venue_patterns:
            matches = re.findall(pattern, text)
            venues.extend(matches)
        
        return venues
    
    def _extract_case_references(self, text):
        """Extract references to legal cases"""
        case_refs = []
        case_patterns = [
            r'Case No\.\s*([^.]+)',
            r'Cause No\.\s*([^.]+)',
            r'in the matter of\s+([^.]+)',
            r'(?:vs?\.|versus)\s+([^.]+)'
        ]
        
        for pattern in case_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            case_refs.extend(matches)
        
        return case_refs
    
    def _extract_legal_captions(self, text):
        """Extract legal document captions"""
        captions = []
        
        # Look for formal legal captions
        caption_patterns = [
            r'IN THE\s+([^§]+)',
            r'BEFORE THE\s+([^§]+)'
        ]
        
        for pattern in caption_patterns:
            matches = re.findall(pattern, text)
            captions.extend(matches)
        
        return captions
    
    def _extract_notary_info(self, text):
        """Extract notary public information"""
        notary_info = []
        
        # Notary name patterns
        notary_patterns = [
            r'Notary Public:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'before me,\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),?\s+(?:a )?[Nn]otary',
        ]
        
        for pattern in notary_patterns:
            matches = re.findall(pattern, text)
            notary_info.extend([f"Notary: {name}" for name in matches])
        
        # Commission expiration
        commission_pattern = r'(?:commission expires|expires)\s+([^.]+)'
        commission_matches = re.findall(commission_pattern, text, re.IGNORECASE)
        notary_info.extend([f"Commission expires: {exp}" for exp in commission_matches])
        
        return notary_info
    
    def _extract_witness_info(self, text):
        """Extract witness information"""
        witnesses = []
        witness_patterns = [
            r'witness(?:es)?:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'in the presence of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        ]
        
        for pattern in witness_patterns:
            matches = re.findall(pattern, text)
            witnesses.extend(matches)
        
        return witnesses
    
    def _detect_seal_references(self, text):
        """Detect references to official seals"""
        seal_indicators = ['seal', 'official seal', 'notarial seal', '[SEAL]', '(SEAL)']
        detected_seals = []
        
        for indicator in seal_indicators:
            if indicator.lower() in text.lower():
                detected_seals.append(indicator.title())
        
        return detected_seals
    
    def _extract_commission_info(self, text):
        """Extract notary commission information"""
        commission_info = []
        
        # Commission number
        commission_pattern = r'commission (?:no\.?|number)\s*:?\s*(\w+)'
        commission_matches = re.findall(commission_pattern, text, re.IGNORECASE)
        commission_info.extend([f"Commission: {comm}" for comm in commission_matches])
        
        return commission_info
    
    def _extract_all_dates(self, text):
        """Extract all dates from text"""
        dates = []
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY
            r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',  # MM-DD-YYYY
            r'\b\w+\s+\d{1,2},?\s+\d{4}\b',  # Month DD, YYYY
            r'\b\d{1,2}\s+\w+\s+\d{4}\b'    # DD Month YYYY
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            dates.extend(matches)
        
        return list(set(dates))
    
    def _analyze_document_structure(self, text):
        """Analyze the structure of the affidavit document"""
        lines = text.strip().split('\n')
        structure = {
            "total_lines": len(lines),
            "non_empty_lines": len([line for line in lines if line.strip()]),
            "sections_detected": self._detect_affidavit_sections(text),
            "formatting_indicators": self._detect_affidavit_formatting(text),
            "formal_elements": self._count_formal_elements(text)
        }
        return structure
    
    def _detect_affidavit_sections(self, text):
        """Detect common affidavit sections"""
        sections = []
        section_headers = [
            'venue', 'oath', 'statement', 'facts', 'verification', 
            'signature', 'notarization', 'jurat'
        ]
        
        for header in section_headers:
            if header.lower() in text.lower():
                sections.append(header.title())
        
        return sections
    
    def _detect_affidavit_formatting(self, text):
        """Detect affidavit-specific formatting"""
        indicators = []
        
        if re.search(r'STATE OF\s+\w+', text, re.IGNORECASE):
            indicators.append("formal_venue")
        if re.search(r'I,\s+[A-Z]', text):
            indicators.append("first_person_statement")
        if re.search(r'_+', text):  # Signature lines
            indicators.append("signature_lines")
        if re.search(r'\d+\.', text):  # Numbered paragraphs
            indicators.append("numbered_paragraphs")
        
        return indicators
    
    def _count_formal_elements(self, text):
        """Count formal legal elements"""
        elements = {
            "oath_phrases": len(self._detect_oath_language(text)),
            "signature_references": len(self._detect_signature_elements(text)),
            "notary_references": len(self._detect_notary_elements(text)),
            "date_references": len(self._extract_all_dates(text)),
            "legal_terms": self._count_legal_terms(text)
        }
        return elements
    
    def _count_legal_terms(self, text):
        """Count legal terminology usage"""
        legal_terms = [
            'affiant', 'deponent', 'sworn', 'affirm', 'solemnly', 'perjury',
            'jurisdiction', 'venue', 'notary', 'acknowledgment', 'jurat'
        ]
        
        count = 0
        for term in legal_terms:
            count += text.lower().count(term.lower())
        
        return count
    
    def _assess_confidence_factors(self, text):
        """Assess factors that affect confidence in extraction"""
        factors = {
            "text_length_adequate": len(text) > 100,
            "contains_formal_elements": any(phrase in text.lower() for phrase in ['state of', 'county of', 'sworn']),
            "contains_affiant_identification": bool(re.search(r'I,\s+[A-Z]', text)),
            "contains_oath_language": bool(self._detect_oath_language(text)),
            "contains_notary_elements": bool(self._detect_notary_elements(text)),
            "contains_dates": bool(self._extract_all_dates(text)),
            "legal_terminology_present": self._count_legal_terms(text) > 0
        }
        return factors

    def _map_to_policy_variable(self, var_name, generic_summary, text_content):
        """Map intelligent analysis results to specific policy variables"""
        # Extract relevant data from generic summary
        affiant_info = generic_summary.get("intelligent_analysis", {}).get("affiant_information", {})
        legal_elements = generic_summary.get("intelligent_analysis", {}).get("legal_elements", {})
        content_analysis = generic_summary.get("intelligent_analysis", {}).get("content_analysis", {})
        formal_elements = generic_summary.get("intelligent_analysis", {}).get("formal_elements", {})
        verification = generic_summary.get("intelligent_analysis", {}).get("verification_elements", {})
        
        # Map based on variable name
        mapping = {
            'DocumentType': lambda: self._determine_document_type(text_content),
            'DocumentDate': lambda: formal_elements.get('document_date', ''),
            'Affiant': lambda: ', '.join(affiant_info.get('names', [])),
            'AffiantTitle': lambda: ', '.join(affiant_info.get('titles', [])),
            'JurisdictionCounty': lambda: self._extract_county_jurisdiction(legal_elements.get('jurisdiction', [])),
            'JurisdictionState': lambda: self._extract_state_jurisdiction(legal_elements.get('jurisdiction', [])),
            'NotaryName': lambda: self._extract_notary_name(verification.get('notary_info', [])),
            'NotaryCommission': lambda: self._extract_commission_number(verification.get('commission_info', [])),
            'Subject': lambda: ', '.join(content_analysis.get('subject_matter', [])),
            'Facts': lambda: self._summarize_facts(content_analysis.get('factual_statements', [])),
            'SwornDate': lambda: formal_elements.get('sworn_date', ''),
            'Purpose': lambda: ', '.join(content_analysis.get('purpose', [])),
            'Attachments': lambda: ', '.join(content_analysis.get('supporting_documents', [])),
            'LegalBasis': lambda: ', '.join(legal_elements.get('legal_basis', []))
        }
        
        if var_name in mapping:
            try:
                return mapping[var_name]()
            except Exception:
                return ""
        
        return ""
    
    def _determine_document_type(self, text):
        """Determine the specific type of document"""
        if 'affidavit' in text.lower():
            return 'Affidavit'
        elif 'declaration' in text.lower():
            return 'Declaration'
        elif 'sworn statement' in text.lower():
            return 'Sworn Statement'
        else:
            return 'Affidavit'  # Default
    
    def _extract_county_jurisdiction(self, jurisdictions):
        """Extract county from jurisdiction list"""
        for jurisdiction in jurisdictions:
            if isinstance(jurisdiction, str) and 'county' in jurisdiction.lower():
                return jurisdiction
        return ""
    
    def _extract_state_jurisdiction(self, jurisdictions):
        """Extract state from jurisdiction list"""
        for jurisdiction in jurisdictions:
            if isinstance(jurisdiction, str) and 'state' in jurisdiction.lower():
                return jurisdiction
        return "Texas"  # Default as per original config
    
    def _extract_notary_name(self, notary_info):
        """Extract notary name from notary information"""
        for info in notary_info:
            if isinstance(info, str) and info.startswith('Notary:'):
                return info.replace('Notary:', '').strip()
        return ""
    
    def _extract_commission_number(self, commission_info):
        """Extract commission number from commission information"""
        for info in commission_info:
            if isinstance(info, str) and info.startswith('Commission:'):
                return info.replace('Commission:', '').strip()
        return ""
    
    def _summarize_facts(self, factual_statements):
        """Summarize factual statements"""
        if not factual_statements:
            return ""
        
        # Take first few statements or summarize if too many
        if len(factual_statements) <= 3:
            return '; '.join(factual_statements)
        else:
            return f"{'; '.join(factual_statements[:3])}... ({len(factual_statements)} total statements)"
    
    def _calculate_field_confidence(self, field_name, extracted_value, text_content):
        """Calculate confidence score for a specific field"""
        if not extracted_value:
            return 0.0
        
        # Base confidence based on extraction success
        base_confidence = 0.5
        
        # Boost confidence based on field-specific factors
        confidence_boosts = {
            'DocumentType': 0.3 if extracted_value != 'Affidavit' else 0.2,  # Higher for specific types
            'Affiant': 0.3 if len(extracted_value.split()) >= 2 else 0.1,  # Full names score higher
            'DocumentDate': 0.3 if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', extracted_value) else 0.1,
            'SwornDate': 0.3 if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', extracted_value) else 0.1,
            'Subject': 0.2 if extracted_value else 0.0,
            'Facts': 0.3 if len(extracted_value) > 50 else 0.1,  # Longer fact descriptions score higher
            'NotaryName': 0.3 if len(extracted_value.split()) >= 2 else 0.1,
            'JurisdictionCounty': 0.2 if 'county' in extracted_value.lower() else 0.1,
            'JurisdictionState': 0.2 if extracted_value != 'Texas' else 0.1  # Non-default values score higher
        }
        
        boost = confidence_boosts.get(field_name, 0.1)
        final_confidence = min(1.0, base_confidence + boost)
        
        return round(final_confidence, 2)
    
    def _calculate_confidence(self, generic_summary, policy_structured_output):
        """Calculate overall confidence score for the processing"""
        # Get individual field confidences
        field_confidences = []
        for field_data in policy_structured_output.values():
            if isinstance(field_data, dict) and 'confidence' in field_data:
                field_confidences.append(field_data['confidence'])
        
        if not field_confidences:
            return 0.5
        
        # Calculate weighted average
        avg_confidence = sum(field_confidences) / len(field_confidences)
        
        # Apply hybrid config weights if available
        hybrid_config = self.variables_config.get('_hybrid_config', {})
        phase_weights = hybrid_config.get('phase_weights', {'intelligent_analysis': 0.65, 'policy_structure': 0.35})
        
        # Factor in intelligent analysis quality
        analysis_quality = self._assess_analysis_quality(generic_summary)
        
        overall_confidence = (
            analysis_quality * phase_weights.get('intelligent_analysis', 0.65) +
            avg_confidence * phase_weights.get('policy_structure', 0.35)
        )
        
        return round(min(1.0, overall_confidence), 2)
    
    def _assess_analysis_quality(self, generic_summary):
        """Assess the quality of the intelligent analysis phase"""
        analysis = generic_summary.get("intelligent_analysis", {})
        confidence_factors = generic_summary.get("extraction_metadata", {}).get("confidence_factors", {})
        
        # Count successful extractions
        successful_extractions = 0
        total_extractions = 0
        
        for category, data in analysis.items():
            if isinstance(data, dict):
                total_extractions += len(data)
                successful_extractions += sum(1 for v in data.values() if v)
        
        if total_extractions == 0:
            return 0.5
        
        extraction_ratio = successful_extractions / total_extractions
        
        # Factor in confidence factors
        confidence_factor_score = sum(1 for factor in confidence_factors.values() if factor) / len(confidence_factors) if confidence_factors else 0.5
        
        # Weighted combination
        return (extraction_ratio * 0.7) + (confidence_factor_score * 0.3)

# Module interface for the pipeline
def get_module():
    """Return the module instance for pipeline integration"""
    return AffidavitModule()

# Metadata for module discovery
MODULE_INFO = {
    "name": "affidavit",
    "version": "2.0.0",
    "description": "Hybrid affidavit document processing module (Python + JSON)",
    "author": "INTV Pipeline",
    "supported_types": ["text", "pdf", "docx"],
    "requires_llm": False,
    "approach": "hybrid_python_json",
    "features": [
        "affiant_identification",
        "legal_element_extraction", 
        "oath_detection",
        "notary_verification",
        "formal_structure_analysis",
        "policy_compliance_structuring"
    ]
}
