"""
Child Module for INTV Pipeline
Handles processing of child-related documents and interactions using hybrid approach
"""

import json
import re
from pathlib import Path

class ChildModule:
    """Module for processing child-related content using hybrid approach"""
    
    def __init__(self):
        self.module_name = "child"
        self.description = "Processes child-related content using hybrid Python + JSON approach"
        self.variables_config = self._load_variables_config()
    def _load_variables_config(self):
        """Load and process the JSON variables configuration"""
        # Look for config in intv/modules directory
        config_paths = [
            Path(__file__).parent.parent / "intv" / "modules" / "child_vars.json",
            Path(__file__).parent / "child_vars.json"
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
                'child_identification': ['name_recognition', 'age_detection', 'role_identification'],
                'safety_assessment': ['risk_indicators', 'protective_factors', 'environmental_safety'],
                'developmental_indicators': ['cognitive_assessment', 'social_skills', 'emotional_indicators'],
                'family_dynamics': ['relationship_mapping', 'household_composition', 'support_systems'],
                'interview_elements': ['rapport_building', 'communication_style', 'cooperation_level']
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
                'type': 'child',
                'label': 'Child Interview',
                'description': 'Default configuration for child interview processing'
            },
            'Name': {'default': '[No Name]', 'hint': 'Full name of the child'},
            'Age': {'default': '', 'hint': 'Age of the child'},
            'Role': {'default': 'Child', 'hint': 'Role in household'},
            'Summary': {'default': 'Child interview processed', 'hint': 'Summary of child interview'},
            '_hybrid_config': {
                'confidence_threshold': 0.7,
                'extraction_strategies': {
                    'child_identification': ['name_recognition'],
                    'safety_assessment': ['risk_indicators']
                }
            }
        }
        
    def process(self, text_content, metadata=None):
        """
        Process child-related content using hybrid approach
        
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
            "content_type": "child",
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
                "child_identification": self._extract_child_info(text_content),
                "safety_assessment": self._assess_safety_indicators(text_content),
                "developmental_indicators": self._analyze_developmental_factors(text_content),
                "family_dynamics": self._analyze_family_dynamics(text_content),
                "interview_elements": self._analyze_interview_elements(text_content),
                "environmental_factors": self._assess_environmental_factors(text_content)
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
    
    def _extract_child_info(self, text):
        """Extract basic child identification information"""
        child_info = {
            "names": self._extract_child_names(text),
            "age_indicators": self._extract_age_info(text),
            "role_identifiers": self._extract_role_info(text),
            "demographic_info": self._extract_demographic_info(text)
        }
        return child_info
    
    def _assess_safety_indicators(self, text):
        """Assess safety-related indicators"""
        safety_assessment = {
            "risk_factors": self._identify_risk_factors(text),
            "protective_factors": self._identify_protective_factors(text),
            "safety_concerns": self._identify_safety_concerns(text),
            "environmental_safety": self._assess_environmental_safety(text)
        }
        return safety_assessment
    
    def _analyze_developmental_factors(self, text):
        """Analyze developmental indicators"""
        developmental = {
            "cognitive_indicators": self._assess_cognitive_level(text),
            "emotional_indicators": self._assess_emotional_state(text),
            "social_skills": self._assess_social_development(text),
            "behavioral_observations": self._identify_behavioral_patterns(text)
        }
        return developmental
    
    def _analyze_family_dynamics(self, text):
        """Analyze family structure and dynamics"""
        family_dynamics = {
            "household_composition": self._identify_household_members(text),
            "family_relationships": self._analyze_family_relationships(text),
            "support_systems": self._identify_support_systems(text),
            "family_functioning": self._assess_family_functioning(text)
        }
        return family_dynamics
    
    def _analyze_interview_elements(self, text):
        """Analyze interview-specific elements"""
        interview_elements = {
            "rapport_building": self._assess_rapport_building(text),
            "communication_style": self._analyze_communication_patterns(text),
            "cooperation_level": self._assess_cooperation(text),
            "interview_setting": self._analyze_interview_setting(text)
        }
        return interview_elements
    
    def _assess_environmental_factors(self, text):
        """Assess environmental and contextual factors"""
        environmental = {
            "living_conditions": self._assess_living_conditions(text),
            "school_environment": self._assess_school_factors(text),
            "community_factors": self._assess_community_factors(text),
            "physical_environment": self._assess_physical_environment(text)
        }
        return environmental
    
    # Child-specific extraction methods
    def _extract_child_names(self, text):
        """Extract child names from text"""
        names = []
        
        # Look for patterns like "The child, [Name]" or "Child's name is [Name]"
        name_patterns = [
            r'[Cc]hild.*name.*is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'[Tt]he child,?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'[Ss]tudent\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'[Mm]inor\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, text)
            names.extend(matches)
        
        return list(set(names))
    
    def _extract_age_info(self, text):
        """Extract age-related information"""
        age_info = []
        
        # Age patterns
        age_patterns = [
            r'(\d+)\s*years?\s*old',
            r'age\s*:?\s*(\d+)',
            r'(\d+)\s*yr\s*old',
            r'born\s+(?:in\s+)?(\d{4})'
        ]
        
        for pattern in age_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            age_info.extend(matches)
        
        return list(set(age_info))
    
    def _extract_role_info(self, text):
        """Extract role information"""
        roles = []
        role_patterns = [
            r'(?:son|daughter|child|student|sibling|oldest|youngest)',
            r'grade\s+(\d+)',
            r'(\w+)\s+child'  # first child, middle child, etc.
        ]
        
        for pattern in role_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            roles.extend(matches)
        
        return list(set(roles))
    
    def _extract_demographic_info(self, text):
        """Extract demographic information"""
        demographic = []
        
        # Look for grade, school, gender indicators
        demo_patterns = [
            r'grade\s+(\d+)',
            r'school\s*:?\s*([A-Z][^.]+)',
            r'attends\s+([A-Z][^.]+)'
        ]
        
        for pattern in demo_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            demographic.extend(matches)
        
        return demographic
    
    def _identify_risk_factors(self, text):
        """Identify risk factors mentioned"""
        risk_factors = []
        risk_keywords = [
            'abuse', 'neglect', 'violence', 'trauma', 'harm', 'danger',
            'unsafe', 'inappropriate', 'concerning', 'worrying'
        ]
        
        for keyword in risk_keywords:
            if keyword.lower() in text.lower():
                risk_factors.append(keyword)
        
        return risk_factors
    
    def _identify_protective_factors(self, text):
        """Identify protective factors"""
        protective_factors = []
        protective_keywords = [
            'safe', 'protected', 'secure', 'stable', 'supportive',
            'caring', 'loving', 'nurturing', 'healthy'
        ]
        
        for keyword in protective_keywords:
            if keyword.lower() in text.lower():
                protective_factors.append(keyword)
        
        return protective_factors
    
    def _identify_safety_concerns(self, text):
        """Identify specific safety concerns"""
        concerns = []
        concern_patterns = [
            r'concerned about\s+([^.]+)',
            r'worry about\s+([^.]+)',
            r'afraid of\s+([^.]+)',
            r'scared of\s+([^.]+)'
        ]
        
        for pattern in concern_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concerns.extend(matches)
        
        return concerns
    
    def _assess_environmental_safety(self, text):
        """Assess environmental safety indicators"""
        safety_indicators = []
        
        if 'clean' in text.lower():
            safety_indicators.append('clean_environment')
        if 'organized' in text.lower():
            safety_indicators.append('organized_space')
        if 'adequate' in text.lower():
            safety_indicators.append('adequate_resources')
        
        return safety_indicators
        family_keywords = ["parent", "guardian", "family", "mother", "father", "sibling"]
        return any(keyword.lower() in text.lower() for keyword in family_keywords)
    
    def _check_developmental_info(self, text):
        """Check for developmental information"""
        dev_keywords = ["development", "school", "education", "behavior", "learning"]
        return any(keyword.lower() in text.lower() for keyword in dev_keywords)
    
    def _analyze_structure(self, text):
        """Basic structure analysis"""
        lines = text.strip().split('\n')
        return {
            "line_count": len(lines),
            "has_multiple_items": len([line for line in lines if line.strip()]) > 1,
            "estimated_items": len([line for line in lines if line.strip()])
        }

# Module interface for the pipeline
def get_module():
    """Return the module instance for pipeline integration"""
    return ChildModule()

# Metadata for module discovery
MODULE_INFO = {
    "name": "child",
    "version": "1.0.0",
    "description": "Child content processing module", 
    "author": "INTV Pipeline",
    "supported_types": ["text", "pdf", "docx", "audio", "video"],
    "requires_llm": False
}
