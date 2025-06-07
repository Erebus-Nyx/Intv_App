#!/usr/bin/env python3
"""
INTV Pipeline CLI - Unified command-line interface for processing different input types
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Optional

from .pipeline_orchestrator import PipelineOrchestrator, InputType
from .config import load_config


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def print_result(result, output_format: str = 'text', output_file: Optional[str] = None):
    """Print or save processing results"""
    
    if output_format == 'json':
        # Convert result to JSON-serializable format
        result_dict = {
            'success': result.success,
            'input_type': result.input_type.value,
            'extracted_text': result.extracted_text,
            'transcript': result.transcript,
            'chunks': result.chunks,
            'segments': result.segments,
            'error_message': result.error_message,
            'metadata': result.metadata,
            'rag_result': result.rag_result,
            'llm_output': result.llm_output
        }
        
        output_text = json.dumps(result_dict, indent=2, ensure_ascii=False)
        
    else:  # text format
        lines = []
        lines.append(f"Processing Result: {'SUCCESS' if result.success else 'FAILED'}")
        lines.append(f"Input Type: {result.input_type.value}")
        
        if result.error_message:
            lines.append(f"Error: {result.error_message}")
        
        if result.extracted_text:
            lines.append(f"\nExtracted Text ({len(result.extracted_text)} chars):")
            lines.append("-" * 50)
            lines.append(result.extracted_text[:1000] + "..." if len(result.extracted_text) > 1000 else result.extracted_text)
        
        if result.chunks:
            lines.append(f"\nChunks: {len(result.chunks)}")
            for i, chunk in enumerate(result.chunks[:3]):  # Show first 3 chunks
                lines.append(f"  Chunk {i+1}: {chunk[:100]}...")
        
        if result.segments:
            lines.append(f"\nTranscript Segments: {len(result.segments)}")
            for i, seg in enumerate(result.segments[:5]):  # Show first 5 segments
                start = seg.get('start', 'N/A')
                end = seg.get('end', 'N/A')
                text = seg.get('text', '')
                lines.append(f"  [{start}s-{end}s]: {text}")
        
        if result.rag_result:
            lines.append(f"\nRAG Result Available: {type(result.rag_result)}")
        
        if result.llm_output:
            lines.append(f"\nLLM Output Available: {type(result.llm_output)}")
        
        if result.metadata:
            lines.append(f"\nMetadata: {json.dumps(result.metadata, indent=2)}")
        
        output_text = "\n".join(lines)
    
    # Output to file or stdout
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"Results saved to: {output_file}")
    else:
        print(output_text)


def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description='INTV Pipeline - Process documents, images, audio files, and microphone input through RAG and LLM pipelines',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a document
  intv-pipeline --files document.pdf --module adult --query "Analyze this content"
  
  # Process an image with OCR
  intv-pipeline --files image.png --module child --apply-llm
  
  # Process audio file
  intv-pipeline --files audio.wav --module ar --diarization
  
  # Record from microphone (start/stop)
  intv-pipeline --microphone --module adult
  
  # Basic processing without module
  intv-pipeline --files document.pdf
  
  # Batch process multiple files
  intv-pipeline --files file1.pdf file2.png file3.wav --module adult --output results.json
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--files', nargs='+', help='Input files to process')
    input_group.add_argument('--microphone', '-m', action='store_true', 
                           help='Record from microphone')
    
    # Microphone options
    mic_group = parser.add_argument_group('Microphone Options')
    mic_group.add_argument('--save-audio', type=str,
                          help='Save recorded audio to file')
    
    # Processing options
    proc_group = parser.add_argument_group('Processing Options')
    proc_group.add_argument('--module', '-M', type=str,
                           help='Module/interview type (e.g., adult, ar, child) - optional for basic processing')
    proc_group.add_argument('--query', '-q', type=str,
                           help='Query for RAG processing (default: "Analyze and summarize this content")')
    proc_group.add_argument('--apply-llm', action='store_true', default=True,
                           help='Apply LLM processing (default: True)')
    proc_group.add_argument('--no-llm', dest='apply_llm', action='store_false',
                           help='Skip LLM processing')
    
    # Audio processing options
    audio_group = parser.add_argument_group('Audio Processing Options')
    audio_group.add_argument('--diarization', action='store_true',
                            help='Enable speaker diarization for audio')
    audio_group.add_argument('--whisper-model', type=str,
                            help='Whisper model to use for transcription')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output', '-o', type=str,
                             help='Output file for results')
    output_group.add_argument('--format', choices=['text', 'json'], default='text',
                             help='Output format (default: text)')
    output_group.add_argument('--verbose', '-v', action='store_true',
                             help='Verbose logging')
    
    # Configuration options
    config_group = parser.add_argument_group('Configuration Options')
    config_group.add_argument('--config', '-c', type=str,
                             help='Path to configuration file')
    config_group.add_argument('--list-modules', action='store_true',
                             help='List available modules and exit')
    
    return parser


def list_available_modules():
    """List available interview modules"""
    try:
        from .module_utils import get_available_interview_types
        modules = get_available_interview_types()
        
        if not modules:
            print("No modules found. Ensure *_vars.json files exist in src/modules/.")
            return
        
        print("Available Interview Modules:")
        print("-" * 30)
        for module in modules:
            key = module.get('key', 'unknown')
            name = module.get('name', 'Unknown')
            description = module.get('description', 'No description available')
            print(f"  {key}: {name}")
            print(f"    {description}")
            print()
            
    except ImportError:
        print("Module utilities not available")
    except Exception as e:
        print(f"Error listing modules: {e}")


def process_single_file(orchestrator: PipelineOrchestrator, 
                       file_path: str,
                       args) -> Optional[object]:
    """Process a single file"""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            return None
        
        print(f"Processing: {file_path}")
        
        # Process the file
        result = orchestrator.process(
            input_path=file_path,
            module_key=args.module,
            query=args.query,
            apply_llm=args.apply_llm
        )
        
        return result
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def process_microphone(orchestrator: PipelineOrchestrator, args) -> Optional[object]:
    """Process microphone input with start/stop control"""
    try:
        print("Starting microphone recording...")
        print("Press Enter to stop recording")
        
        # Process microphone input
        result = orchestrator.process_microphone(
            module_key=args.module,
            query=args.query,
            output_path=args.save_audio,
            apply_llm=args.apply_llm
        )
        
        return result
        
    except KeyboardInterrupt:
        print("\nRecording stopped by user")
        return None
    except Exception as e:
        print(f"Error processing microphone input: {e}")
        return None


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # List modules if requested
    if args.list_modules:
        list_available_modules()
        return
    
    # Load configuration
    try:
        config = load_config()
        
        # Update config with CLI arguments
        if args.diarization:
            config['enable_diarization'] = True
        if args.whisper_model:
            config['whisper_model'] = args.whisper_model
            
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Create pipeline orchestrator
    try:
        orchestrator = PipelineOrchestrator(args.config)
    except Exception as e:
        print(f"Error initializing pipeline orchestrator: {e}")
        sys.exit(1)
    
    # Process inputs
    results = []
    
    if args.microphone:
        # Process microphone input
        result = process_microphone(orchestrator, args)
        if result:
            results.append(result)
    
    elif args.files:
        # Process file inputs
        for file_path in args.files:
            result = process_single_file(orchestrator, file_path, args)
            if result:
                results.append(result)
    
    else:
        print("No input specified. Use --help for usage information.")
        sys.exit(1)
    
    # Output results
    if results:
        if len(results) == 1:
            print_result(results[0], args.format, args.output)
        else:
            # Multiple results
            for i, result in enumerate(results):
                if args.output:
                    # Generate unique output filenames
                    output_path = Path(args.output)
                    stem = output_path.stem
                    suffix = output_path.suffix
                    output_file = f"{stem}_{i+1}{suffix}"
                else:
                    output_file = None
                
                print(f"\n{'='*60}")
                print(f"Result {i+1}/{len(results)}")
                print(f"{'='*60}")
                print_result(result, args.format, output_file)
    else:
        print("No results to display.")
        sys.exit(1)


if __name__ == "__main__":
    main()
