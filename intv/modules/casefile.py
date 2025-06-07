"""
Casefile Module for INTV Pipeline - Hybrid Approach
Combines Python-based intelligent analysis with JSON-based policy constraints
"""

import json
import os
import re
from pathlib import Path

class CasefileModule:
    """Module for processing casefile-related content using hybrid approach"""
    
    def __init__(self):
        self.module_name = "casefile"
        self.description = "Processes casefile documents using hybrid Python + JSON approach"
        self.variables_config = self._load_variables_config()
    def _load_variables_config(self):
        """Load and process the JSON variables configuration"""
        # Look for config in intv/modules directory
        config_paths = [
            Path(__file__).parent.parent / "intv" / "modules" / "casefile_vars.json",
            Path(__file__).parent / "casefile_vars.json"
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
            'confidence_threshold': 0.7,
            'extraction_strategies': {
                'case_identifiers': ['regex_patterns', 'keyword_matching'],
                'dates': ['regex_patterns', 'nlp_extraction'],
                'participants': ['name_recognition', 'role_identification'],
                'legal_elements': ['keyword_matching', 'pattern_recognition']
            },
            'phase_weights': {
                'intelligent_analysis': 0.6,
                'policy_structure': 0.4
            }
        }
        return hybrid_config
    
    def _get_default_config(self):
        """Return default configuration if no file found"""
        return {
            '_header': {
                'type': 'casefile',
                'label': 'Case File Document',
                'description': 'Default configuration for casefile processing'
            },
            'CaseNumber': {'default': '', 'hint': 'Official case number or identifier'},
            'CaseType': {'default': '', 'hint': 'Type of case'},
            'Summary': {'default': 'Case file analysis completed', 'hint': 'Summary of case file contents'},
            '_hybrid_config': {
                'confidence_threshold': 0.7,
                'extraction_strategies': {
                    'case_identifiers': ['regex_patterns', 'keyword_matching'],
                    'dates': ['regex_patterns'],
                    'participants': ['name_recognition'],
                    'legal_elements': ['keyword_matching']
                }
            }
        }
        
    def process(self, text_content, metadata=None):
        """
        Process casefile content using hybrid approach
        
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
            "content_type": "casefile",
            "processing_phases": {
                "phase_1": "python_intelligent_analysis", 
                "phase_2": "json_policy_constraints"
            },
            "generic_summary": generic_summary,
            "policy_structured_output": policy_structured_output,
            "confidence_score": confidence,
            "metadata": metadata or {}
        }
    
    def _create_generic_summary(self, text_content, metadata=None):
        """
        Phase 1: Python-based intelligent analysis
        Extract meaningful information using intelligent Python processing
        """
        summary = {
            "intelligent_analysis": {
                "case_information": self._extract_case_information(text_content),
                "participant_analysis": self._extract_participants(text_content),
                "timeline_analysis": self._extract_timeline(text_content),
                "legal_elements": self._extract_legal_elements(text_content),
                "document_structure": self._analyze_document_structure(text_content),
                "content_classification": self._classify_content(text_content)
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
    
    def _extract_case_information(self, text):
        """Extract case-specific information"""
        case_info = {
            "case_numbers": self._find_case_numbers(text),
            "case_types": self._identify_case_types(text),
            "jurisdictions": self._extract_jurisdictions(text),
            "official_identifiers": self._find_official_identifiers(text)
        }
        return case_info
    
    def _extract_participants(self, text):
        """Extract information about case participants"""
        participants = {
            "identified_names": self._extract_names(text),
            "roles_and_titles": self._extract_roles(text),
            "caseworkers": self._identify_caseworkers(text),
            "legal_professionals": self._identify_legal_professionals(text),
            "family_members": self._identify_family_members(text)
        }
        return participants
    
    def _extract_timeline(self, text):
        """Extract temporal information and case timeline"""
        timeline = {
            "dates_mentioned": self._extract_dates(text),
            "chronological_events": self._identify_events(text),
            "case_milestones": self._identify_milestones(text),
            "temporal_relationships": self._analyze_temporal_relationships(text)
        }
        return timeline
    
    def _extract_legal_elements(self, text):
        """Extract legal elements and case-specific details"""
        legal_elements = {
            "allegations": self._extract_allegations(text),
            "findings": self._extract_findings(text),
            "court_orders": self._extract_court_orders(text),
            "legal_actions": self._extract_legal_actions(text),
            "outcomes": self._extract_outcomes(text)
        }
        return legal_elements
    
    def _find_case_numbers(self, text):
        """Find case numbers and official identifiers"""
        case_numbers = []
        patterns = [
            r'\bcase\s*#?\s*(\d{4,})',  # "case #123" or "case 123"
            r'\bcause\s*#?\s*(\d{4,})',  # "cause #123"
            r'\bdocket\s*#?\s*(\d{4,})',  # "docket #123"
            r'\b(\d{4}-\d{4,})\b',  # "2023-1234" format
            r'\b(\d{8,})\b'  # 8+ digit numbers
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            case_numbers.extend(matches)
        
        return list(set(case_numbers))  # Remove duplicates
    
    def _identify_case_types(self, text):
        """Identify the type of case based on content"""
        case_types = []
        type_indicators = {
            'CPS': ['child protective', 'cps', 'dfps', 'family services'],
            'Family Court': ['family court', 'custody', 'visitation', 'divorce'],
            'Criminal': ['criminal', 'charges', 'arrest', 'prosecution'],
            'Civil': ['civil suit', 'lawsuit', 'plaintiff', 'defendant'],
            'Juvenile': ['juvenile', 'minor', 'youth court'],
            'Adoption': ['adoption', 'termination of parental rights', 'tpr']
        }
        
        for case_type, keywords in type_indicators.items():
            if any(keyword.lower() in text.lower() for keyword in keywords):
                case_types.append(case_type)
        
        return case_types
    
    def _extract_jurisdictions(self, text):
        """Extract jurisdiction information"""
        jurisdictions = []
        
        # Look for county references
        county_pattern = r'(\w+)\s+county'
        counties = re.findall(county_pattern, text, re.IGNORECASE)
        jurisdictions.extend([f"{county} County" for county in counties])
        
        # Look for court references
        court_pattern = r'(\d+(?:st|nd|rd|th)?)\s+(?:district|judicial)\s+court'
        courts = re.findall(court_pattern, text, re.IGNORECASE)
        jurisdictions.extend([f"{court} District Court" for court in courts])
        
        return list(set(jurisdictions))
    
    def _find_official_identifiers(self, text):
        """Find various official identifiers"""
        identifiers = []
        
        # Social Security Numbers (partially redacted)
        ssn_pattern = r'xxx-xx-\d{4}|\*\*\*-\*\*-\d{4}'
        ssn_matches = re.findall(ssn_pattern, text)
        identifiers.extend([f"SSN: {ssn}" for ssn in ssn_matches])
        
        # Driver's License patterns
        dl_pattern = r'(?:DL|driver.?s license)[:\s]*(\w+\d+)'
        dl_matches = re.findall(dl_pattern, text, re.IGNORECASE)
        identifiers.extend([f"DL: {dl}" for dl in dl_matches])
        
        return identifiers
    
    def _extract_names(self, text):
        """Extract names of people mentioned"""
        names = []
        
        # Look for capitalized name patterns
        name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        potential_names = re.findall(name_pattern, text)
        
        # Filter out common non-names
        excluded_words = {'Texas', 'County', 'Court', 'Department', 'Services', 'Case', 'File'}
        names = [name for name in potential_names 
                if not any(word in name for word in excluded_words) and len(name.split()) <= 3]
        
        return list(set(names))
    
    def _extract_roles(self, text):
        """Extract roles and titles mentioned"""
        roles = []
        role_patterns = [
            r'(\w+(?:\s+\w+)*)\s+caseworker',
            r'judge\s+(\w+(?:\s+\w+)*)',
            r'attorney\s+(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+supervisor'
        ]
        
        for pattern in role_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            roles.extend(matches)
        
        return list(set(roles))
    
    def _identify_caseworkers(self, text):
        """Identify caseworkers mentioned"""
        caseworkers = []
        cw_keywords = ['caseworker', 'case worker', 'social worker', 'investigator']
        
        for keyword in cw_keywords:
            if keyword.lower() in text.lower():
                # Try to find names near caseworker mentions
                keyword_positions = [m.start() for m in re.finditer(keyword, text, re.IGNORECASE)]
                for pos in keyword_positions:
                    # Look for names within 50 characters before the keyword
                    context = text[max(0, pos-50):pos+50]
                    names = self._extract_names(context)
                    caseworkers.extend(names)
        
        return list(set(caseworkers))
    
    def _identify_legal_professionals(self, text):
        """Identify legal professionals mentioned"""
        legal_pros = []
        legal_keywords = ['attorney', 'lawyer', 'judge', 'magistrate', 'prosecutor']
        
        for keyword in legal_keywords:
            if keyword.lower() in text.lower():
                keyword_positions = [m.start() for m in re.finditer(keyword, text, re.IGNORECASE)]
                for pos in keyword_positions:
                    context = text[max(0, pos-30):pos+50]
                    names = self._extract_names(context)
                    legal_pros.extend([f"{name} ({keyword})" for name in names])
        
        return list(set(legal_pros))
    
    def _identify_family_members(self, text):
        """Identify family members and relationships"""
        family_members = []
        family_keywords = ['mother', 'father', 'parent', 'child', 'children', 'sibling', 'guardian']
        
        for keyword in family_keywords:
            if keyword.lower() in text.lower():
                keyword_positions = [m.start() for m in re.finditer(keyword, text, re.IGNORECASE)]
                for pos in keyword_positions:
                    context = text[max(0, pos-30):pos+50]
                    names = self._extract_names(context)
                    family_members.extend([f"{name} ({keyword})" for name in names])
        
        return list(set(family_members))
    
    def _extract_dates(self, text):
        """Extract dates mentioned in the text"""
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
    
    def _identify_events(self, text):
        """Identify chronological events in the case"""
        events = []
        event_keywords = ['investigation', 'interview', 'visit', 'hearing', 'court', 'removal', 'placement']
        
        for keyword in event_keywords:
            if keyword.lower() in text.lower():
                events.append(keyword.title())
        
        return events
    
    def _identify_milestones(self, text):
        """Identify case milestones"""
        milestones = []
        milestone_keywords = ['opened', 'closed', 'substantiated', 'unsubstantiated', 'court order', 'placement']
        
        for keyword in milestone_keywords:
            if keyword.lower() in text.lower():
                milestones.append(keyword.title())
        
        return milestones
    
    def _analyze_temporal_relationships(self, text):
        """Analyze temporal relationships between events"""
        temporal_indicators = ['before', 'after', 'during', 'since', 'until', 'following', 'prior to']
        relationships = []
        
        for indicator in temporal_indicators:
            if indicator.lower() in text.lower():
                relationships.append(indicator)
        
        return relationships
    
    def _extract_allegations(self, text):
        """Extract allegations mentioned"""
        allegations = []
        allegation_keywords = ['neglect', 'abuse', 'physical abuse', 'sexual abuse', 'emotional abuse', 
                              'medical neglect', 'educational neglect', 'abandonment']
        
        for keyword in allegation_keywords:
            if keyword.lower() in text.lower():
                allegations.append(keyword.title())
        
        return allegations
    
    def _extract_findings(self, text):
        """Extract investigation findings"""
        findings = []
        finding_keywords = ['substantiated', 'unsubstantiated', 'inconclusive', 'ruled out', 
                           'confirmed', 'unable to determine']
        
        for keyword in finding_keywords:
            if keyword.lower() in text.lower():
                findings.append(keyword.title())
        
        return findings
    
    def _extract_court_orders(self, text):
        """Extract court orders mentioned"""
        orders = []
        order_keywords = ['temporary restraining order', 'protective order', 'removal order', 
                         'service plan', 'court order', 'judicial order']
        
        for keyword in order_keywords:
            if keyword.lower() in text.lower():
                orders.append(keyword.title())
        
        return orders
    
    def _extract_legal_actions(self, text):
        """Extract legal actions taken"""
        actions = []
        action_keywords = ['filed petition', 'emergency removal', 'court hearing', 'mediation', 
                          'trial', 'appeal', 'motion filed']
        
        for keyword in action_keywords:
            if keyword.lower() in text.lower():
                actions.append(keyword.title())
        
        return actions
    
    def _extract_outcomes(self, text):
        """Extract case outcomes"""
        outcomes = []
        outcome_keywords = ['case closed', 'services completed', 'children returned', 'parental rights terminated', 
                           'adoption finalized', 'case transferred']
        
        for keyword in outcome_keywords:
            if keyword.lower() in text.lower():
                outcomes.append(keyword.title())
        
        return outcomes
    
    def _analyze_document_structure(self, text):
        """Analyze the structure of the document"""
        lines = text.strip().split('\n')
        structure = {
            "total_lines": len(lines),
            "non_empty_lines": len([line for line in lines if line.strip()]),
            "sections_detected": self._detect_sections(text),
            "formatting_indicators": self._detect_formatting(text)
        }
        return structure
    
    def _detect_sections(self, text):
        """Detect document sections"""
        sections = []
        section_headers = ['background', 'allegations', 'investigation', 'findings', 'recommendations', 
                          'court orders', 'services', 'outcome', 'summary']
        
        for header in section_headers:
            if header.lower() in text.lower():
                sections.append(header.title())
        
        return sections
    
    def _detect_formatting(self, text):
        """Detect formatting indicators"""
        indicators = []
        
        if re.search(r'\d+\.', text):  # Numbered lists
            indicators.append("numbered_lists")
        if re.search(r'^\s*[-*]', text, re.MULTILINE):  # Bullet points
            indicators.append("bullet_points")
        if re.search(r'^[A-Z][^a-z]*:', text, re.MULTILINE):  # Headers with colons
            indicators.append("section_headers")
        
        return indicators
    
    def _classify_content(self, text):
        """Classify the type of casefile content"""
        classifications = []
        
        content_types = {
            'Investigation Report': ['investigation', 'allegation', 'interview', 'finding'],
            'Court Filing': ['petition', 'motion', 'order', 'hearing'],
            'Case Summary': ['summary', 'overview', 'history', 'timeline'],
            'Service Plan': ['service plan', 'goal', 'objective', 'task'],
            'Progress Report': ['progress', 'update', 'status', 'completion']
        }
        
        for content_type, keywords in content_types.items():
            if any(keyword.lower() in text.lower() for keyword in keywords):
                classifications.append(content_type)
        
        return classifications
    
    def _assess_confidence_factors(self, text):
        """Assess factors that affect confidence in extraction"""
        factors = {
            "text_length_adequate": len(text) > 100,
            "contains_structured_data": bool(re.search(r'\d+', text)),
            "contains_proper_names": bool(re.search(r'\b[A-Z][a-z]+\b', text)),
            "contains_dates": bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)),
            "contains_legal_language": any(word in text.lower() for word in ['court', 'case', 'allegation', 'finding'])
        }
        return factors
    
    def _map_to_policy_variable(self, var_name, generic_summary, text_content):
        """Map intelligent analysis results to specific policy variables"""
        # Extract relevant data from generic summary
        case_info = generic_summary.get("intelligent_analysis", {}).get("case_information", {})
        participants = generic_summary.get("intelligent_analysis", {}).get("participant_analysis", {})
        timeline = generic_summary.get("intelligent_analysis", {}).get("timeline_analysis", {})
        legal_elements = generic_summary.get("intelligent_analysis", {}).get("legal_elements", {})
        
        # Map based on variable name
        mapping = {
            'CaseNumber': lambda: ', '.join(case_info.get('case_numbers', [])),
            'CaseType': lambda: ', '.join(case_info.get('case_types', [])),
            'OpenDate': lambda: timeline.get('dates_mentioned', [''])[0] if timeline.get('dates_mentioned') else '',
            'CloseDate': lambda: self._find_close_date(timeline.get('dates_mentioned', [])),
            'CaseWorker': lambda: ', '.join(participants.get('caseworkers', [])),
            'County': lambda: ', '.join([j for j in case_info.get('jurisdictions', []) if 'County' in j]),
            'Court': lambda: ', '.join([j for j in case_info.get('jurisdictions', []) if 'Court' in j]),
            'Judge': lambda: ', '.join([p for p in participants.get('legal_professionals', []) if 'judge' in p.lower()]),
            'Participants': lambda: ', '.join(participants.get('identified_names', [])),
            'Children': lambda: ', '.join([p for p in participants.get('family_members', []) if 'child' in p.lower()]),
            'Adults': lambda: ', '.join([p for p in participants.get('family_members', []) if any(role in p.lower() for role in ['mother', 'father', 'parent'])]),
            'Allegations': lambda: ', '.join(legal_elements.get('allegations', [])),
            'Findings': lambda: ', '.join(legal_elements.get('findings', [])),
            'Services': lambda: self._extract_services_mentioned(text_content),
            'Outcome': lambda: ', '.join(legal_elements.get('outcomes', [])),
            'SafetyPlan': lambda: self._extract_safety_plan(text_content),
            'Removals': lambda: self._extract_removals(text_content),
            'CourtOrders': lambda: ', '.join(legal_elements.get('court_orders', [])),
            'NextSteps': lambda: self._extract_next_steps(text_content)
        }
        
        if var_name in mapping:
            try:
                return mapping[var_name]()
            except Exception:
                return ""
        
        return ""
    
    def _find_close_date(self, dates):
        """Find the most likely close date from extracted dates"""
        # Simple heuristic: if multiple dates, the last one might be close date
        if len(dates) > 1:
            return dates[-1]
        return ""
    
    def _extract_services_mentioned(self, text):
        """Extract services mentioned in the text"""
        services = []
        service_keywords = ['counseling', 'therapy', 'parenting classes', 'substance abuse treatment', 
                           'mental health services', 'family preservation', 'supervised visitation']
        
        for keyword in service_keywords:
            if keyword.lower() in text.lower():
                services.append(keyword.title())
        
        return ', '.join(services)
    
    def _extract_safety_plan(self, text):
        """Extract safety plan information"""
        if 'safety plan' in text.lower():
            return "Safety plan implemented"
        return ""
    
    def _extract_removals(self, text):
        """Extract removal information"""
        removals = []
        if 'removal' in text.lower():
            removals.append("Child removal documented")
        if 'placement' in text.lower():
            removals.append("Placement arranged")
        
        return ', '.join(removals)
    
    def _extract_next_steps(self, text):
        """Extract next steps or follow-up actions"""
        next_steps = []
        step_keywords = ['follow-up', 'next visit', 'court date', 'review', 'monitoring']
        
        for keyword in step_keywords:
            if keyword.lower() in text.lower():
                next_steps.append(keyword.title())
        
        return ', '.join(next_steps)
    
    def _calculate_field_confidence(self, field_name, extracted_value, text_content):
        """Calculate confidence score for a specific field"""
        if not extracted_value:
            return 0.0
        
        # Base confidence based on extraction success
        base_confidence = 0.5
        
        # Boost confidence based on field-specific factors
        confidence_boosts = {
            'CaseNumber': 0.3 if re.search(r'\d{4,}', extracted_value) else 0.1,
            'CaseType': 0.2 if extracted_value else 0.0,
            'OpenDate': 0.3 if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', extracted_value) else 0.1,
            'CaseWorker': 0.2 if len(extracted_value.split()) >= 2 else 0.1,
            'Allegations': 0.3 if extracted_value else 0.0,
            'Findings': 0.3 if extracted_value else 0.0
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
        phase_weights = hybrid_config.get('phase_weights', {'intelligent_analysis': 0.6, 'policy_structure': 0.4})
        
        # Factor in intelligent analysis quality
        analysis_quality = self._assess_analysis_quality(generic_summary)
        
        overall_confidence = (
            analysis_quality * phase_weights.get('intelligent_analysis', 0.6) +
            avg_confidence * phase_weights.get('policy_structure', 0.4)
        )
        
        return round(min(1.0, overall_confidence), 2)
    
    def _assess_analysis_quality(self, generic_summary):
        """Assess the quality of the intelligent analysis phase"""
        analysis = generic_summary.get("intelligent_analysis", {})
        
        # Count successful extractions
        successful_extractions = 0
        total_extractions = 0
        
        for category, data in analysis.items():
            if isinstance(data, dict):
                total_extractions += len(data)
                successful_extractions += sum(1 for v in data.values() if v)
        
        if total_extractions == 0:
            return 0.5
        
        return successful_extractions / total_extractions

# Module interface for the pipeline
def get_module():
    """Return the module instance for pipeline integration"""
    return CasefileModule()

# Metadata for module discovery
MODULE_INFO = {
    "name": "casefile", 
    "version": "2.0.0",
    "description": "Hybrid casefile document processing module (Python + JSON)",
    "author": "INTV Pipeline",
    "supported_types": ["text", "pdf", "docx"],
    "requires_llm": False,
    "approach": "hybrid_python_json",
    "features": [
        "intelligent_case_analysis",
        "participant_extraction", 
        "timeline_analysis",
        "legal_element_identification",
        "policy_compliance_structuring"
    ]
}
