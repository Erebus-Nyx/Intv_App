# Medical Intake Module

## Overview
- **Module ID**: medical_intake
- **Domain**: medical
- **Content Type**: form
- **Purpose**: Extract patient information and medical history from intake forms

## Description
Processes medical intake forms to extract patient demographics, medical history, and current symptoms

## Policy Variables

### patient_name
- **Type**: string
- **Required**: True
- **Default**: None
- **Description**: Full name of the patient

### date_of_birth
- **Type**: date
- **Required**: True
- **Default**: None
- **Description**: Patient's date of birth

### chief_complaint
- **Type**: text
- **Required**: True
- **Default**: None
- **Description**: Primary reason for visit or main symptom

### medical_history
- **Type**: text
- **Required**: False
- **Default**: None reported
- **Description**: Past medical conditions, surgeries, medications

### allergies
- **Type**: text
- **Required**: False
- **Default**: No known allergies
- **Description**: Known allergies or adverse reactions

### current_medications
- **Type**: text
- **Required**: False
- **Default**: None
- **Description**: Current medications and dosages

### emergency_contact
- **Type**: string
- **Required**: True
- **Default**: None
- **Description**: Name and phone number of emergency contact

### insurance_info
- **Type**: string
- **Required**: False
- **Default**: 
- **Description**: Insurance provider and member ID


## Usage
This module was created using the Universal Module Creator system and can process any content
related to medical in the form format.

### Integration with INTV Pipeline
```python
from intv.modules.enhanced_dynamic_module import enhanced_dynamic_module_output

result = enhanced_dynamic_module_output(
    text_content="your content here",
    module_key="medical_intake",
    output_path="output.json"
)
```

## Configuration Files
- `medical_intake_vars.json` - Main module configuration
- `medical_intake_strategies.json` - Extraction strategies  
- `medical_intake_mappings.json` - Policy mappings
- `medical_intake_example_config.yaml` - Example user configuration

## Customization
You can modify the module behavior by editing the configuration files or creating
a new version using the Universal Module Creator with updated parameters.

---
*Generated by Universal Module Creator on 2025-06-08 14:20:09*
