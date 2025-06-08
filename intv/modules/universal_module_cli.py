"""
Command-Line Interface for Universal Module Creator
Provides interactive module creation and management capabilities.
"""

import argparse
import json
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from .universal_module_creator import UniversalModuleCreator, create_universal_module, EXAMPLE_CONFIGURATIONS

class UniversalModuleCLI:
    """Command-line interface for Universal Module Creator"""
    
    def __init__(self):
        self.creator = UniversalModuleCreator()
    
    def run(self, args: list = None):
        """Run the CLI with provided arguments"""
        if args is None:
            args = sys.argv[1:]
        
        parser = self._create_parser()
        parsed_args = parser.parse_args(args)
        
        if hasattr(parsed_args, 'func'):
            try:
                result = parsed_args.func(parsed_args)
                if isinstance(result, dict):
                    self._print_result(result)
                return result
            except Exception as e:
                print(f"Error: {str(e)}", file=sys.stderr)
                return {"success": False, "error": str(e)}
        else:
            parser.print_help()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser"""
        parser = argparse.ArgumentParser(
            description="Universal Module Creator for INTV Framework",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Create module from configuration file
  python -m intv.modules.universal_module_cli create --config my_module.yaml
  
  # Create module interactively
  python -m intv.modules.universal_module_cli create --interactive
  
  # List available modules
  python -m intv.modules.universal_module_cli list
  
  # Generate example configuration
  python -m intv.modules.universal_module_cli example --domain legal --output legal_example.yaml
  
  # Test a created module
  python -m intv.modules.universal_module_cli test --module my_module --content "test content"
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Create command
        create_parser = subparsers.add_parser('create', help='Create a new universal module')
        create_group = create_parser.add_mutually_exclusive_group(required=True)
        create_group.add_argument('--config', '-c', help='Configuration file (YAML or JSON)')
        create_group.add_argument('--interactive', '-i', action='store_true', help='Interactive module creation')
        create_parser.set_defaults(func=self._create_module)
        
        # List command
        list_parser = subparsers.add_parser('list', help='List existing modules')
        list_parser.add_argument('--detailed', '-d', action='store_true', help='Show detailed information')
        list_parser.set_defaults(func=self._list_modules)
        
        # Example command
        example_parser = subparsers.add_parser('example', help='Generate example configurations')
        example_parser.add_argument('--domain', choices=['legal', 'medical', 'business'], 
                                  help='Domain for example configuration')
        example_parser.add_argument('--output', '-o', help='Output file for example configuration')
        example_parser.add_argument('--list-examples', action='store_true', help='List available examples')
        example_parser.set_defaults(func=self._generate_example)
        
        # Test command
        test_parser = subparsers.add_parser('test', help='Test a created module')
        test_parser.add_argument('--module', '-m', required=True, help='Module ID to test')
        test_parser.add_argument('--content', '-c', help='Test content (text)')
        test_parser.add_argument('--file', '-f', help='Test content from file')
        test_parser.set_defaults(func=self._test_module)
        
        # Validate command
        validate_parser = subparsers.add_parser('validate', help='Validate a module configuration')
        validate_parser.add_argument('config', help='Configuration file to validate')
        validate_parser.set_defaults(func=self._validate_config)
        
        # Info command
        info_parser = subparsers.add_parser('info', help='Show information about a module')
        info_parser.add_argument('module_id', help='Module ID to show info for')
        info_parser.set_defaults(func=self._show_module_info)
        
        return parser
    
    def _create_module(self, args) -> Dict[str, Any]:
        """Create a new module"""
        if args.interactive:
            return self._interactive_module_creation()
        else:
            return create_universal_module(config_file_path=args.config)
    
    def _interactive_module_creation(self) -> Dict[str, Any]:
        """Interactive module creation wizard"""
        print("🚀 Universal Module Creator - Interactive Mode")
        print("=" * 50)
        
        try:
            # Basic information
            module_id = input("Module ID (snake_case): ").strip()
            if not module_id.isidentifier():
                return {"success": False, "error": "Module ID must be a valid identifier"}
            
            label = input(f"Module Label [{module_id.replace('_', ' ').title()}]: ").strip()
            if not label:
                label = module_id.replace('_', ' ').title()
            
            # Context
            print("\n📋 Context Information:")
            purpose = input("Purpose (what this module should accomplish): ").strip()
            domain = input("Domain (legal/medical/business/research/education/other): ").strip()
            content_type = input("Content Type (interview/report/transcript/form/document): ").strip()
            description = input("Description (optional): ").strip()
            
            # Policy Structure
            print("\n📝 Policy Structure (Variables):")
            print("Enter variable names one by one. Press Enter with empty name to finish.")
            
            policy_structure = {}
            while True:
                var_name = input("\nVariable name (or Enter to finish): ").strip()
                if not var_name:
                    break
                
                hint = input(f"  Hint for {var_name}: ").strip()
                var_type = input(f"  Type for {var_name} [string]: ").strip() or "string"
                default = input(f"  Default value for {var_name}: ").strip()
                required_input = input(f"  Required? (y/N): ").strip().lower()
                required = required_input in ['y', 'yes', 'true']
                
                policy_structure[var_name] = {
                    "hint": hint,
                    "type": var_type,
                    "default": default,
                    "required": required
                }
            
            if not policy_structure:
                return {"success": False, "error": "At least one policy variable is required"}
            
            # Build configuration
            config = {
                "module_id": module_id,
                "label": label,
                "context": {
                    "purpose": purpose,
                    "domain": domain,
                    "content_type": content_type,
                    "description": description
                },
                "policy_structure": policy_structure
            }
            
            # Confirmation
            print(f"\n✅ Module Configuration Summary:")
            print(f"  Module ID: {module_id}")
            print(f"  Domain: {domain}")
            print(f"  Variables: {len(policy_structure)}")
            
            confirm = input("\nCreate this module? (Y/n): ").strip().lower()
            if confirm in ['n', 'no']:
                return {"success": False, "error": "Module creation cancelled by user"}
            
            # Create the module
            return self.creator.create_module_from_config(config)
            
        except KeyboardInterrupt:
            return {"success": False, "error": "Module creation cancelled"}
        except Exception as e:
            return {"success": False, "error": f"Interactive creation failed: {str(e)}"}
    
    def _list_modules(self, args) -> Dict[str, Any]:
        """List existing modules"""
        try:
            modules_dir = self.creator.modules_dir
            module_files = list(modules_dir.glob("*_vars.json"))
            
            modules = []
            for file_path in module_files:
                module_name = file_path.stem.replace("_vars", "")
                
                if args.detailed:
                    try:
                        with open(file_path, 'r') as f:
                            config = json.load(f)
                        
                        header = config.get("_header", {})
                        var_count = len([k for k in config.keys() if not k.startswith("_")])
                        
                        modules.append({
                            "id": module_name,
                            "label": header.get("label", module_name),
                            "domain": header.get("domain", "unknown"),
                            "variables": var_count,
                            "created": header.get("created", "unknown"),
                            "framework": header.get("framework", "legacy")
                        })
                    except:
                        modules.append({
                            "id": module_name,
                            "label": module_name,
                            "domain": "unknown",
                            "variables": "unknown",
                            "created": "unknown",
                            "framework": "legacy"
                        })
                else:
                    modules.append({"id": module_name})
            
            return {
                "success": True,
                "modules": modules,
                "total_count": len(modules)
            }
            
        except Exception as e:
            return {"success": False, "error": f"Failed to list modules: {str(e)}"}
    
    def _generate_example(self, args) -> Dict[str, Any]:
        """Generate example configurations"""
        try:
            if args.list_examples:
                examples = []
                for key, config in EXAMPLE_CONFIGURATIONS.items():
                    examples.append({
                        "key": key,
                        "module_id": config["module_id"],
                        "domain": config["context"]["domain"],
                        "description": config["context"]["purpose"]
                    })
                return {"success": True, "examples": examples}
            
            if args.domain:
                # Find example for domain
                example_config = None
                for key, config in EXAMPLE_CONFIGURATIONS.items():
                    if config["context"]["domain"].lower() == args.domain.lower():
                        example_config = config
                        break
                
                if not example_config:
                    return {"success": False, "error": f"No example found for domain: {args.domain}"}
                
                # Save to file if requested
                if args.output:
                    output_path = Path(args.output)
                    with open(output_path, 'w') as f:
                        yaml.dump(example_config, f, default_flow_style=False, indent=2)
                    
                    return {
                        "success": True,
                        "example_saved": str(output_path),
                        "domain": args.domain,
                        "config": example_config
                    }
                else:
                    return {
                        "success": True,
                        "domain": args.domain,
                        "config": example_config
                    }
            
            # Show all examples
            return {
                "success": True,
                "available_examples": list(EXAMPLE_CONFIGURATIONS.keys()),
                "use_--domain": "Specify --domain to get a specific example"
            }
            
        except Exception as e:
            return {"success": False, "error": f"Failed to generate example: {str(e)}"}
    
    def _test_module(self, args) -> Dict[str, Any]:
        """Test a created module"""
        try:
            # Get test content
            if args.content:
                test_content = args.content
            elif args.file:
                file_path = Path(args.file)
                if not file_path.exists():
                    return {"success": False, "error": f"Test file not found: {args.file}"}
                with open(file_path, 'r', encoding='utf-8') as f:
                    test_content = f.read()
            else:
                test_content = f"This is a test document for the {args.module} module."
            
            # Import and test
            from .enhanced_dynamic_module import enhanced_dynamic_module_output
            
            result = enhanced_dynamic_module_output(
                text_content=test_content,
                module_key=args.module
            )
            
            return {
                "success": True,
                "module_id": args.module,
                "test_content_length": len(test_content),
                "processing_result": result,
                "variables_extracted": len(result.get("policy_structured_output", {})),
                "confidence_score": result.get("confidence_score", 0)
            }
            
        except Exception as e:
            return {"success": False, "error": f"Module test failed: {str(e)}"}
    
    def _validate_config(self, args) -> Dict[str, Any]:
        """Validate a module configuration"""
        try:
            config_path = Path(args.config)
            if not config_path.exists():
                return {"success": False, "error": f"Configuration file not found: {args.config}"}
            
            # Load configuration
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                with open(config_path, 'r') as f:
                    config = json.load(f)
            
            # Validate
            validation_result = self.creator._validate_config(config)
            
            return {
                "success": True,
                "config_file": str(config_path),
                "validation": validation_result
            }
            
        except Exception as e:
            return {"success": False, "error": f"Validation failed: {str(e)}"}
    
    def _show_module_info(self, args) -> Dict[str, Any]:
        """Show information about a specific module"""
        try:
            module_id = args.module_id
            modules_dir = self.creator.modules_dir
            
            # Check if module exists
            config_file = modules_dir / f"{module_id}_vars.json"
            if not config_file.exists():
                return {"success": False, "error": f"Module {module_id} not found"}
            
            # Load configuration
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Get module information
            header = config.get("_header", {})
            variables = {k: v for k, v in config.items() if not k.startswith("_")}
            
            # Check for related files
            related_files = []
            for suffix in ["_strategies.json", "_mappings.json", "_README.md", "_example_config.yaml"]:
                file_path = modules_dir / f"{module_id}{suffix}"
                if file_path.exists():
                    related_files.append(file_path.name)
            
            return {
                "success": True,
                "module_id": module_id,
                "header": header,
                "variables": variables,
                "variable_count": len(variables),
                "related_files": related_files,
                "config_file": config_file.name
            }
            
        except Exception as e:
            return {"success": False, "error": f"Failed to get module info: {str(e)}"}
    
    def _print_result(self, result: Dict[str, Any]):
        """Print formatted result"""
        if result.get("success"):
            print("✅ Success!")
            
            # Print specific result details
            if "module_id" in result:
                print(f"   Module ID: {result['module_id']}")
            
            if "files_created" in result:
                print(f"   Files created: {len(result['files_created'])}")
                for file_path in result["files_created"]:
                    print(f"     - {Path(file_path).name}")
            
            if "modules" in result:
                print(f"   Found {result['total_count']} modules:")
                for module in result["modules"]:
                    if isinstance(module, dict) and "label" in module:
                        print(f"     - {module['id']}: {module['label']} ({module['domain']})")
                    else:
                        print(f"     - {module['id'] if isinstance(module, dict) else module}")
            
            if "examples" in result:
                print("   Available examples:")
                for example in result["examples"]:
                    print(f"     - {example['key']}: {example['description']}")
            
            if "config" in result and "examples" not in result:
                print("   Configuration:")
                config = result["config"]
                if isinstance(config, dict) and "context" in config:
                    print(f"     Domain: {config['context'].get('domain', 'unknown')}")
                    print(f"     Purpose: {config['context'].get('purpose', 'unknown')}")
            
            if "validation" in result:
                validation = result["validation"]
                print(f"   Valid: {validation['valid']}")
                if validation.get("errors"):
                    print("   Errors:")
                    for error in validation["errors"]:
                        print(f"     - {error}")
                if validation.get("warnings"):
                    print("   Warnings:")
                    for warning in validation["warnings"]:
                        print(f"     - {warning}")
            
            if "processing_result" in result:
                proc_result = result["processing_result"]
                print(f"   Variables extracted: {result.get('variables_extracted', 0)}")
                print(f"   Confidence score: {result.get('confidence_score', 0):.2f}")
                
        else:
            print("❌ Error!")
            print(f"   {result.get('error', 'Unknown error')}")

def main():
    """Main entry point for CLI"""
    cli = UniversalModuleCLI()
    result = cli.run()
    if isinstance(result, dict) and not result.get("success"):
        sys.exit(1)

if __name__ == "__main__":
    main()
