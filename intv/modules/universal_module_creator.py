"""
Universal Module Creator for INTV Framework
Creates configuration-driven modules for any domain or application without code modification.

This system allows users to create custom modules by providing:
1. Context description (purpose, scope, content type)
2. Policy structure (variables, constraints, output format)  
3. Extraction strategies (patterns, methods, confidence levels)

The system automatically generates extraction strategies and policy mappings
for any domain: legal, medical, business, research, education, etc.
"""

import json
import os
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

class UniversalModuleCreator:
    """Creates domain-agnostic modules from user configuration"""
    
    def __init__(self, modules_dir: str = None):
        """
        Initialize the universal module creator
        
        Args:
            modules_dir: Directory to store module configurations
        """
        self.modules_dir = Path(modules_dir) if modules_dir else Path(__file__).parent
        self.template_dir = self.modules_dir / "templates"
        self.template_dir.mkdir(exist_ok=True)
        
        # Initialize universal extraction patterns
        self.universal_patterns = self._load_universal_patterns()
        
        # Initialize domain templates
        self.domain_templates = self._load_domain_templates()
    
    def create_module_from_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new module from user configuration
        
        Args:
            config: Universal module configuration with context, policy, and extraction info
            
        Returns:
            Dict with created module information and validation results
        """
        try:
            # Validate configuration
            validation_result = self._validate_config(config)
            if not validation_result["valid"]:
                return {"success": False, "error": validation_result["errors"]}
            
            # Generate module configuration
            module_config = self._generate_module_config(config)
            
            # Create extraction strategies
            extraction_strategies = self._create_extraction_strategies(config)
            
            # Generate policy mappings
            policy_mappings = self._generate_policy_mappings(config)
            
            # Create module files
            module_files = self._create_module_files(
                config, module_config, extraction_strategies, policy_mappings
            )
            
            # Test the created module
            test_result = self._test_created_module(config["module_id"])
            
            return {
                "success": True,
                "module_id": config["module_id"],
                "files_created": module_files,
                "config": module_config,
                "test_result": test_result,
                "creation_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"success": False, "error": f"Module creation failed: {str(e)}"}
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate universal module configuration"""
        errors = []
        warnings = []
        
        # Required fields
        required_fields = ["module_id", "context", "policy_structure"]
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Validate module_id
        if "module_id" in config:
            module_id = config["module_id"]
            if not isinstance(module_id, str) or not module_id.isidentifier():
                errors.append("module_id must be a valid Python identifier")
            if (self.modules_dir / f"{module_id}_vars.json").exists():
                warnings.append(f"Module {module_id} already exists and will be overwritten")
        
        # Validate context
        if "context" in config:
            context = config["context"]
            required_context_fields = ["purpose", "domain", "content_type"]
            for field in required_context_fields:
                if field not in context:
                    errors.append(f"Missing context field: {field}")
        
        # Validate policy structure
        if "policy_structure" in config:
            policy = config["policy_structure"]
            if not isinstance(policy, dict) or not policy:
                errors.append("policy_structure must be a non-empty dictionary")
            
            # Validate policy variables
            for var_name, var_config in policy.items():
                if not isinstance(var_config, dict):
                    errors.append(f"Policy variable {var_name} must be a dictionary")
                    continue
                    
                # Check for hint (recommended)
                if "hint" not in var_config:
                    warnings.append(f"Policy variable {var_name} lacks a hint")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def _generate_module_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complete module configuration from user input"""
        module_config = {
            "_header": {
                "type": config["module_id"],
                "label": config.get("label", config["module_id"].replace("_", " ").title()),
                "description": config["context"]["purpose"],
                "domain": config["context"]["domain"],
                "content_type": config["context"]["content_type"],
                "created": datetime.now().isoformat(),
                "version": "1.0",
                "framework": "universal_module_creator"
            },
            "_extraction_config": {
                "methods": config.get("extraction_strategies", ["smart_patterns", "keyword_search"]),
                "confidence_threshold": config.get("confidence_threshold", 0.7),
                "patterns": self._generate_domain_patterns(config["context"]["domain"]),
                "keywords": self._generate_domain_keywords(config["context"])
            },
            "_processing_config": {
                "fallback_behavior": config.get("fallback_behavior", "prompt_user"),
                "auto_fill": config.get("auto_fill", True),
                "validation_rules": config.get("validation_rules", []),
                "output_format": config.get("output_format", "structured_json")
            }
        }
        
        # Add policy structure variables
        for var_name, var_config in config["policy_structure"].items():
            module_config[var_name] = {
                "default": var_config.get("default", ""),
                "hint": var_config.get("hint", f"Information about {var_name.lower()}"),
                "type": var_config.get("type", "string"),
                "required": var_config.get("required", False),
                "validation": var_config.get("validation", {}),
                "extraction_priority": var_config.get("extraction_priority", 5),
                "patterns": var_config.get("patterns", [])
            }
        
        return module_config
    
    def _create_extraction_strategies(self, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """Create domain-specific extraction strategies"""
        domain = config["context"]["domain"].lower()
        content_type = config["context"]["content_type"].lower()
        
        # Base strategies for all domains
        strategies = {
            "universal": [
                "pattern_matching",
                "keyword_extraction", 
                "sentence_analysis",
                "context_inference"
            ],
            "domain_specific": self._get_domain_strategies(domain),
            "content_specific": self._get_content_strategies(content_type)
        }
        
        return strategies
    
    def _generate_policy_mappings(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mappings between extracted data and policy variables"""
        mappings = {}
        
        for var_name, var_config in config["policy_structure"].items():
            mappings[var_name] = {
                "extraction_methods": self._determine_extraction_methods(var_name, var_config),
                "fallback_sources": self._determine_fallback_sources(var_name, var_config),
                "validation_rules": self._create_validation_rules(var_name, var_config),
                "confidence_factors": self._determine_confidence_factors(var_name, var_config)
            }
        
        return mappings
    
    def _create_module_files(self, config: Dict[str, Any], module_config: Dict[str, Any], 
                           extraction_strategies: Dict[str, Any], policy_mappings: Dict[str, Any]) -> List[str]:
        """Create the actual module files"""
        module_id = config["module_id"]
        files_created = []
        
        # 1. Create main module configuration file
        config_file = self.modules_dir / f"{module_id}_vars.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(module_config, f, indent=2, ensure_ascii=False)
        files_created.append(str(config_file))
        
        # 2. Create extraction strategies file
        strategies_file = self.modules_dir / f"{module_id}_strategies.json"
        with open(strategies_file, 'w', encoding='utf-8') as f:
            json.dump(extraction_strategies, f, indent=2, ensure_ascii=False)
        files_created.append(str(strategies_file))
        
        # 3. Create policy mappings file  
        mappings_file = self.modules_dir / f"{module_id}_mappings.json"
        with open(mappings_file, 'w', encoding='utf-8') as f:
            json.dump(policy_mappings, f, indent=2, ensure_ascii=False)
        files_created.append(str(mappings_file))
        
        # 4. Create module documentation
        docs_file = self.modules_dir / f"{module_id}_README.md"
        with open(docs_file, 'w', encoding='utf-8') as f:
            f.write(self._generate_module_documentation(config, module_config))
        files_created.append(str(docs_file))
        
        # 5. Create example configuration (for users to modify)
        example_file = self.modules_dir / f"{module_id}_example_config.yaml"
        with open(example_file, 'w', encoding='utf-8') as f:
            yaml.dump(self._create_example_config(config), f, default_flow_style=False)
        files_created.append(str(example_file))
        
        return files_created
    
    def _load_universal_patterns(self) -> Dict[str, List[str]]:
        """Load universal extraction patterns for common data types"""
        return {
            "names": [
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)\s+(?:is|was|stated|said|reported)',
                r'(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
                r'(?:Name|Patient|Client|Subject):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)',
            ],
            "dates": [
                r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b',
                r'\b([A-Za-z]+\s+\d{1,2},?\s+\d{4})\b',
                r'\b(\d{1,2}\s+[A-Za-z]+\s+\d{4})\b'
            ],
            "ages": [
                r'(?:aged|age)\s+(\d+)',
                r'(\d+)\s*(?:years?\s*old|y\.?o\.?)',
                r'\b(\d{1,2})\s*(?:year|yr)'
            ],
            "addresses": [
                r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln)',
                r'[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}',
                r'\d+\s+[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*[A-Z]{2}'
            ],
            "phone_numbers": [
                r'\b(\d{3}[-.]?\d{3}[-.]?\d{4})\b',
                r'\((\d{3})\)\s*(\d{3})[-.](\d{4})'
            ],
            "emails": [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ]
        }
    
    def _load_domain_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load templates for common domains"""
        return {
            "legal": {
                "common_variables": ["case_number", "court", "judge", "attorney", "incident_date", "parties"],
                "patterns": ["case", "court", "plaintiff", "defendant", "allegation", "evidence"],
                "extraction_priorities": {"case_number": 10, "incident_date": 9, "parties": 8}
            },
            "medical": {
                "common_variables": ["patient_name", "dob", "diagnosis", "symptoms", "treatment", "provider"],
                "patterns": ["patient", "diagnosis", "symptoms", "treatment", "medication", "doctor"],
                "extraction_priorities": {"patient_name": 10, "diagnosis": 9, "symptoms": 8}
            },
            "business": {
                "common_variables": ["company", "employee", "date", "project", "budget", "deliverables"],
                "patterns": ["company", "project", "budget", "deadline", "deliverable", "stakeholder"],
                "extraction_priorities": {"company": 9, "project": 8, "budget": 7}
            },
            "research": {
                "common_variables": ["study_title", "participants", "methodology", "findings", "conclusion"],
                "patterns": ["study", "research", "participants", "methodology", "results", "conclusion"],
                "extraction_priorities": {"study_title": 10, "methodology": 9, "findings": 8}
            },
            "education": {
                "common_variables": ["student_name", "grade", "subject", "performance", "behavior", "goals"],
                "patterns": ["student", "grade", "subject", "performance", "behavior", "achievement"],
                "extraction_priorities": {"student_name": 10, "grade": 9, "subject": 8}
            }
        }
    
    def _generate_domain_patterns(self, domain: str) -> List[str]:
        """Generate domain-specific extraction patterns"""
        domain_template = self.domain_templates.get(domain.lower(), {})
        return domain_template.get("patterns", [])
    
    def _generate_domain_keywords(self, context: Dict[str, Any]) -> List[str]:
        """Generate domain-specific keywords for extraction"""
        domain = context["domain"].lower()
        purpose = context["purpose"].lower()
        
        # Extract keywords from purpose description
        purpose_keywords = [word for word in purpose.split() if len(word) > 3]
        
        # Add domain-specific keywords
        domain_keywords = []
        if domain in self.domain_templates:
            domain_keywords = self.domain_templates[domain].get("patterns", [])
        
        return list(set(purpose_keywords + domain_keywords))
    
    def _get_domain_strategies(self, domain: str) -> List[str]:
        """Get domain-specific extraction strategies"""
        strategies_map = {
            "legal": ["case_analysis", "legal_entity_recognition", "citation_extraction"],
            "medical": ["medical_entity_recognition", "symptom_extraction", "treatment_analysis"],
            "business": ["financial_analysis", "project_extraction", "stakeholder_identification"],
            "research": ["methodology_extraction", "data_analysis", "conclusion_identification"],
            "education": ["academic_performance_analysis", "behavioral_assessment", "goal_identification"]
        }
        return strategies_map.get(domain, ["general_analysis"])
    
    def _get_content_strategies(self, content_type: str) -> List[str]:
        """Get content-type-specific extraction strategies"""
        strategies_map = {
            "interview": ["dialogue_analysis", "speaker_identification", "topic_tracking"],
            "report": ["section_analysis", "structured_extraction", "summary_generation"],
            "transcript": ["temporal_analysis", "speaker_diarization", "topic_segmentation"],
            "form": ["field_extraction", "structured_parsing", "validation_checking"],
            "narrative": ["story_analysis", "event_extraction", "character_identification"]
        }
        return strategies_map.get(content_type, ["text_analysis"])
    
    def _determine_extraction_methods(self, var_name: str, var_config: Dict[str, Any]) -> List[str]:
        """Determine best extraction methods for a specific variable"""
        var_type = var_config.get("type", "string").lower()
        var_name_lower = var_name.lower()
        
        methods = ["keyword_search", "pattern_matching"]
        
        # Add type-specific methods
        if var_type in ["date", "datetime"]:
            methods.extend(["date_extraction", "temporal_analysis"])
        elif var_type in ["number", "integer", "float"]:
            methods.extend(["numeric_extraction", "calculation_analysis"])
        elif "name" in var_name_lower:
            methods.extend(["name_extraction", "entity_recognition"])
        elif "address" in var_name_lower:
            methods.extend(["address_extraction", "location_analysis"])
        elif "phone" in var_name_lower or "contact" in var_name_lower:
            methods.extend(["contact_extraction"])
        
        return methods
    
    def _determine_fallback_sources(self, var_name: str, var_config: Dict[str, Any]) -> List[str]:
        """Determine fallback data sources for a variable"""
        fallbacks = ["user_prompt"]
        
        if var_config.get("default"):
            fallbacks.insert(0, "default_value")
        
        # Add intelligent fallbacks based on variable name
        var_name_lower = var_name.lower()
        if "date" in var_name_lower:
            fallbacks.insert(0, "document_date")
        elif "name" in var_name_lower:
            fallbacks.insert(0, "document_title")
        
        return fallbacks
    
    def _create_validation_rules(self, var_name: str, var_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create validation rules for a variable"""
        rules = []
        
        var_type = var_config.get("type", "string").lower()
        
        # Type-based validation
        if var_type == "date":
            rules.append({
                "type": "date_format",
                "message": "Must be a valid date"
            })
        elif var_type in ["number", "integer", "float"]:
            rules.append({
                "type": "numeric",
                "message": "Must be a valid number"
            })
        elif var_type == "email":
            rules.append({
                "type": "email_format",
                "message": "Must be a valid email address"
            })
        
        # Required field validation
        if var_config.get("required", False):
            rules.append({
                "type": "required",
                "message": f"{var_name} is required"
            })
        
        # Custom validation rules from config
        if "validation" in var_config:
            rules.extend(var_config["validation"])
        
        return rules
    
    def _determine_confidence_factors(self, var_name: str, var_config: Dict[str, Any]) -> Dict[str, float]:
        """Determine confidence calculation factors for a variable"""
        factors = {
            "exact_match": 1.0,
            "pattern_match": 0.8,
            "keyword_proximity": 0.6,
            "context_inference": 0.4,
            "default_fallback": 0.1
        }
        
        # Adjust factors based on variable importance
        priority = var_config.get("extraction_priority", 5)
        multiplier = priority / 10.0
        
        return {key: value * multiplier for key, value in factors.items()}
    
    def _generate_module_documentation(self, config: Dict[str, Any], module_config: Dict[str, Any]) -> str:
        """Generate documentation for the created module"""
        module_id = config["module_id"]
        context = config["context"]
        
        doc = f"""# {module_config['_header']['label']} Module

## Overview
- **Module ID**: {module_id}
- **Domain**: {context['domain']}
- **Content Type**: {context['content_type']}
- **Purpose**: {context['purpose']}

## Description
{context.get('description', 'Auto-generated module for universal content processing.')}

## Policy Variables
"""
        
        for var_name, var_config in config["policy_structure"].items():
            doc += f"\n### {var_name}\n"
            doc += f"- **Type**: {var_config.get('type', 'string')}\n"
            doc += f"- **Required**: {var_config.get('required', False)}\n"
            doc += f"- **Default**: {var_config.get('default', 'None')}\n"
            doc += f"- **Description**: {var_config.get('hint', 'No description provided')}\n"
        
        doc += f"""

## Usage
This module was created using the Universal Module Creator system and can process any content
related to {context['domain']} in the {context['content_type']} format.

### Integration with INTV Pipeline
```python
from intv.modules.enhanced_dynamic_module import enhanced_dynamic_module_output

result = enhanced_dynamic_module_output(
    text_content="your content here",
    module_key="{module_id}",
    output_path="output.json"
)
```

## Configuration Files
- `{module_id}_vars.json` - Main module configuration
- `{module_id}_strategies.json` - Extraction strategies  
- `{module_id}_mappings.json` - Policy mappings
- `{module_id}_example_config.yaml` - Example user configuration

## Customization
You can modify the module behavior by editing the configuration files or creating
a new version using the Universal Module Creator with updated parameters.

---
*Generated by Universal Module Creator on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return doc
    
    def _create_example_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create an example configuration for users to modify"""
        return {
            "module_id": f"{config['module_id']}_custom",
            "context": {
                "purpose": "Modify this to describe your specific use case",
                "domain": config["context"]["domain"], 
                "content_type": config["context"]["content_type"],
                "description": "Detailed description of what this module should accomplish"
            },
            "policy_structure": {
                var_name: {
                    "default": var_config.get("default", ""),
                    "hint": "Modify this hint to guide users",
                    "type": var_config.get("type", "string"),
                    "required": var_config.get("required", False)
                }
                for var_name, var_config in config["policy_structure"].items()
            },
            "extraction_strategies": ["smart_patterns", "keyword_search", "context_inference"],
            "confidence_threshold": 0.7,
            "fallback_behavior": "prompt_user",
            "auto_fill": True
        }
    
    def _test_created_module(self, module_id: str) -> Dict[str, Any]:
        """Test the created module with sample content"""
        try:
            # Import the enhanced dynamic module
            from .enhanced_dynamic_module import enhanced_dynamic_module_output
            
            # Test with minimal content
            test_content = f"This is a test document for the {module_id} module created by the Universal Module Creator."
            
            result = enhanced_dynamic_module_output(
                text_content=test_content,
                module_key=module_id
            )
            
            return {
                "success": True,
                "test_content_length": len(test_content),
                "variables_processed": len(result.get("policy_structured_output", {})),
                "confidence_score": result.get("confidence_score", 0),
                "processing_time": "< 1 second"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Module test failed: {str(e)}"
            }

def create_universal_module(config_file_path: str = None, config_dict: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Convenience function to create a universal module
    
    Args:
        config_file_path: Path to YAML/JSON configuration file
        config_dict: Configuration dictionary (alternative to file)
        
    Returns:
        Dict with creation result
    """
    creator = UniversalModuleCreator()
    
    if config_file_path:
        # Load configuration from file
        config_path = Path(config_file_path)
        if not config_path.exists():
            return {"success": False, "error": f"Configuration file not found: {config_file_path}"}
        
        try:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    if config is None:
                        return {"success": False, "error": "Configuration file is empty or invalid"}
            else:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if config is None:
                        return {"success": False, "error": "Configuration file is empty or invalid"}
        except Exception as e:
            return {"success": False, "error": f"Failed to load configuration: {str(e)}"}
    
    elif config_dict:
        config = config_dict
    else:
        return {"success": False, "error": "Either config_file_path or config_dict must be provided"}
    
    return creator.create_module_from_config(config)

# Example configurations for different domains
EXAMPLE_CONFIGURATIONS = {
    "legal_case_analysis": {
        "module_id": "legal_case",
        "context": {
            "purpose": "Analyze legal case documents to extract key case information and details",
            "domain": "legal",
            "content_type": "case_document",
            "description": "Processes legal case files to extract parties, charges, evidence, and outcomes"
        },
        "policy_structure": {
            "case_number": {"default": "", "hint": "Official case number or docket number", "type": "string", "required": True},
            "court": {"default": "", "hint": "Court name and jurisdiction", "type": "string"},
            "judge": {"default": "", "hint": "Presiding judge name", "type": "string"},
            "plaintiff": {"default": "", "hint": "Plaintiff or prosecutor name", "type": "string"},
            "defendant": {"default": "", "hint": "Defendant name", "type": "string"},
            "charges": {"default": "", "hint": "List of charges or claims", "type": "text"},
            "incident_date": {"default": "", "hint": "Date of alleged incident", "type": "date"},
            "case_status": {"default": "", "hint": "Current status of the case", "type": "string"},
            "summary": {"default": "", "hint": "Brief summary of the case", "type": "text"}
        }
    },
    
    "medical_patient_intake": {
        "module_id": "medical_intake", 
        "context": {
            "purpose": "Process patient intake forms and medical histories",
            "domain": "medical",
            "content_type": "patient_record",
            "description": "Extracts patient information, medical history, and current health status"
        },
        "policy_structure": {
            "patient_name": {"default": "", "hint": "Patient full name", "type": "string", "required": True},
            "date_of_birth": {"default": "", "hint": "Patient date of birth", "type": "date", "required": True},
            "medical_record_number": {"default": "", "hint": "Unique medical record identifier", "type": "string"},
            "chief_complaint": {"default": "", "hint": "Primary reason for visit", "type": "text"},
            "current_medications": {"default": "", "hint": "List of current medications", "type": "text"},
            "allergies": {"default": "", "hint": "Known allergies and reactions", "type": "text"},
            "medical_history": {"default": "", "hint": "Relevant past medical history", "type": "text"},
            "provider": {"default": "", "hint": "Attending physician or provider", "type": "string"},
            "visit_date": {"default": "", "hint": "Date of medical visit", "type": "date"}
        }
    },
    
    "business_meeting_analysis": {
        "module_id": "business_meeting",
        "context": {
            "purpose": "Analyze business meeting transcripts to extract decisions and action items",
            "domain": "business",
            "content_type": "meeting_transcript",
            "description": "Processes meeting notes to identify decisions, action items, and attendees"
        },
        "policy_structure": {
            "meeting_title": {"default": "", "hint": "Title or subject of the meeting", "type": "string"},
            "meeting_date": {"default": "", "hint": "Date of the meeting", "type": "date"},
            "attendees": {"default": "", "hint": "List of meeting attendees", "type": "text"},
            "agenda_items": {"default": "", "hint": "Main topics discussed", "type": "text"},
            "decisions_made": {"default": "", "hint": "Key decisions reached", "type": "text"},
            "action_items": {"default": "", "hint": "Action items and assignments", "type": "text"},
            "next_meeting": {"default": "", "hint": "Date of next meeting if scheduled", "type": "date"},
            "meeting_outcome": {"default": "", "hint": "Overall outcome or conclusion", "type": "text"}
        }
    }
}
