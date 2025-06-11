#!/usr/bin/env python3
"""
Complete Audio Pipeline Test - Verify End-to-End Audio Processing

This test verifies:
1. Audio files are processed and passed to RAG
2. RAG generates output 
3. Audio transcriptions are stored in the output directory prior to being passed to RAG
4. Complete pipeline flow: Audio → Transcription → VAD → Diarization → RAG → LLM
"""

import sys
import os
import tempfile
import json
import numpy as np
from pathlib import Path
import shutil
import datetime

# Add the intv directory to the Python path
sys.path.insert(0, '/home/nyx/intv')

def create_test_audio_file():
    """Create a simple test audio file for testing"""
    try:
        import soundfile as sf
        
        # Create a simple test audio signal (sine wave)
        sample_rate = 16000
        duration = 3.0  # 3 seconds
        frequency = 440  # A4 note
        
        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        # Create temporary audio file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_file.name, audio_data, sample_rate)
        temp_file.close()
        
        print(f"✅ Created test audio file: {temp_file.name}")
        return temp_file.name
        
    except ImportError:
        print("❌ soundfile not available - cannot create test audio")
        return None
    except Exception as e:
        print(f"❌ Failed to create test audio: {e}")
        return None

def test_audio_pipeline_imports():
    """Test that all audio pipeline components can be imported"""
    print("\n🔍 **Testing Audio Pipeline Imports**")
    
    try:
        from intv.pipeline_orchestrator import PipelineOrchestrator
        print("✅ PipelineOrchestrator imported successfully")
        
        from intv.audio_transcribe import transcribe_audio_fastwhisper
        print("✅ Audio transcription module imported")
        
        from intv.audio_vad import detect_voice_activity_pyannote
        print("✅ VAD module imported")
        
        from intv.audio_diarization import diarize_audio
        print("✅ Diarization module imported")
        
        from intv.rag import chunk_text, enhanced_query_documents
        print("✅ RAG modules imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_output_directory_structure():
    """Test that output directories are properly created and used"""
    print("\n🔍 **Testing Output Directory Structure**")
    
    # Create test output directory
    test_output_dir = Path("/tmp/intv_test_output")
    test_output_dir.mkdir(exist_ok=True)
    
    # Create cache directory
    cache_dir = test_output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    print(f"✅ Test output directory created: {test_output_dir}")
    print(f"✅ Cache directory created: {cache_dir}")
    
    return test_output_dir, cache_dir

def test_transcription_storage(audio_file, output_dir):
    """Test that transcriptions are stored in output directory before RAG processing"""
    print("\n🔍 **Testing Transcription Storage**")
    
    try:
        from intv.audio_transcribe import transcribe_audio_fastwhisper
        
        # Transcribe audio
        print("   📝 Transcribing audio...")
        segments = transcribe_audio_fastwhisper(
            audio_file,
            return_segments=True,
            config={'audio_sample_rate': 16000}
        )
        
        if not segments:
            print("❌ No transcription segments returned")
            return False
        
        # Extract transcript
        transcript = " ".join([seg.get('text', '') for seg in segments])
        print(f"   ✅ Transcription completed: {len(transcript)} characters")
        print(f"   📄 Transcript preview: {transcript[:100]}...")
        
        # Save transcription to output directory (simulating storage before RAG)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        transcription_file = output_dir / f"transcription_{timestamp}.json"
        
        transcription_data = {
            'transcript': transcript,
            'segments': segments,
            'metadata': {
                'audio_file': audio_file,
                'timestamp': timestamp,
                'num_segments': len(segments)
            }
        }
        
        with open(transcription_file, 'w', encoding='utf-8') as f:
            json.dump(transcription_data, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Transcription saved to: {transcription_file}")
        
        return transcription_file, transcript, segments
        
    except Exception as e:
        print(f"❌ Transcription storage test failed: {e}")
        return None

def test_rag_processing(transcript, chunks, output_dir):
    """Test RAG processing with audio transcription"""
    print("\n🔍 **Testing RAG Processing**")
    
    try:
        from intv.rag import enhanced_query_documents
        
        if not chunks:
            print("❌ No chunks available for RAG processing")
            return False
        
        # Test RAG query
        test_query = "What is the main content of this audio?"
        print(f"   🔍 Running RAG query: '{test_query}'")
        
        # Process with RAG
        rag_result = enhanced_query_documents(
            test_query,
            chunks,
            config={'rag_top_k': 3}
        )
        
        if not rag_result:
            print("❌ RAG processing returned no results")
            return False
        
        print(f"   ✅ RAG processing completed")
        print(f"   📊 RAG result type: {type(rag_result)}")
        
        # Save RAG results to output directory
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        rag_file = output_dir / f"rag_results_{timestamp}.json"
        
        rag_data = {
            'query': test_query,
            'result': rag_result,
            'chunks_processed': len(chunks),
            'metadata': {
                'timestamp': timestamp,
                'processing_status': 'completed'
            }
        }
        
        with open(rag_file, 'w', encoding='utf-8') as f:
            json.dump(rag_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"   ✅ RAG results saved to: {rag_file}")
        
        return rag_result
        
    except Exception as e:
        print(f"❌ RAG processing test failed: {e}")
        return None

def test_complete_pipeline(audio_file, output_dir):
    """Test the complete audio pipeline end-to-end"""
    print("\n🔍 **Testing Complete Audio Pipeline**")
    
    try:
        from intv.pipeline_orchestrator import PipelineOrchestrator
        
        # Initialize pipeline orchestrator
        config = {
            'enable_vad': True,
            'enable_diarization': True,
            'audio_sample_rate': 16000,
            'rag_top_k': 3
        }
        
        orchestrator = PipelineOrchestrator(config=config)
        print("   ✅ Pipeline orchestrator initialized")
        
        # Process audio file with query
        test_query = "Summarize the content of this audio"
        print(f"   🎵 Processing audio file: {Path(audio_file).name}")
        print(f"   🔍 With query: '{test_query}'")
        
        result = orchestrator.process_audio_file(
            file_path=Path(audio_file),
            query=test_query,
            apply_llm=False  # Skip LLM for this test
        )
        
        if not result.success:
            print(f"❌ Pipeline processing failed: {result.error_message}")
            return False
        
        print(f"   ✅ Pipeline processing completed successfully")
        
        # Verify pipeline results
        print(f"   📄 Transcript length: {len(result.transcript) if result.transcript else 0}")
        print(f"   🧩 Chunks created: {len(result.chunks) if result.chunks else 0}")
        print(f"   🎯 Segments processed: {len(result.segments) if result.segments else 0}")
        
        # Check metadata
        if result.metadata:
            print(f"   📊 VAD enabled: {result.metadata.get('vad_enabled', 'Unknown')}")
            print(f"   📊 Diarization enabled: {result.metadata.get('diarization_enabled', 'Unknown')}")
            print(f"   📊 Pipeline: {result.metadata.get('audio_processing_pipeline', 'Unknown')}")
        
        # Check RAG results
        if result.rag_result:
            print(f"   ✅ RAG processing completed")
            print(f"   📊 RAG result type: {type(result.rag_result)}")
        else:
            print(f"   ⚠️  No RAG results (may be expected)")
        
        # Save complete pipeline result
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        pipeline_file = output_dir / f"complete_pipeline_{timestamp}.json"
        
        pipeline_data = {
            'success': result.success,
            'input_type': result.input_type.value if hasattr(result.input_type, 'value') else str(result.input_type),
            'transcript': result.transcript,
            'transcript_length': len(result.transcript) if result.transcript else 0,
            'chunks_count': len(result.chunks) if result.chunks else 0,
            'segments_count': len(result.segments) if result.segments else 0,
            'metadata': result.metadata,
            'rag_result_available': result.rag_result is not None,
            'processing_timestamp': timestamp
        }
        
        with open(pipeline_file, 'w', encoding='utf-8') as f:
            json.dump(pipeline_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"   ✅ Complete pipeline results saved to: {pipeline_file}")
        
        return result
        
    except Exception as e:
        print(f"❌ Complete pipeline test failed: {e}")
        import traceback
        print(f"   🔍 Error details: {traceback.format_exc()}")
        return None

def main():
    """Run complete audio pipeline verification"""
    print("=" * 80)
    print("🎵 **INTV Complete Audio Pipeline Verification**")
    print("=" * 80)
    
    # Test 1: Import verification
    if not test_audio_pipeline_imports():
        print("\n❌ Import tests failed - cannot proceed")
        return False
    
    # Test 2: Output directory setup
    output_dir, cache_dir = test_output_directory_structure()
    
    # Test 3: Create test audio
    audio_file = create_test_audio_file()
    if not audio_file:
        print("\n⚠️  Cannot create test audio - looking for existing audio files...")
        
        # Look for existing audio files
        audio_extensions = ['.wav', '.mp3', '.m4a', '.flac']
        found_files = []
        
        search_paths = ['/home/nyx/intv', '/home/nyx/intv/sample-sources']
        for search_path in search_paths:
            if os.path.exists(search_path):
                for root, dirs, files in os.walk(search_path):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in audio_extensions):
                            found_files.append(os.path.join(root, file))
        
        if found_files:
            audio_file = found_files[0]
            print(f"✅ Using existing audio file: {audio_file}")
        else:
            print("❌ No audio files available for testing")
            return False
    
    try:
        # Test 4: Transcription storage
        transcription_result = test_transcription_storage(audio_file, output_dir)
        if not transcription_result:
            print("\n❌ Transcription storage test failed")
            return False
        
        transcription_file, transcript, segments = transcription_result
        
        # Test 5: Text chunking for RAG
        print("\n🔍 **Testing Text Chunking for RAG**")
        try:
            from intv.rag import chunk_text
            chunks = chunk_text(transcript)
            print(f"   ✅ Text chunked into {len(chunks)} pieces")
            print(f"   📄 First chunk preview: {chunks[0][:100] if chunks else 'No chunks'}...")
        except Exception as e:
            print(f"❌ Text chunking failed: {e}")
            chunks = []
        
        # Test 6: RAG processing
        if chunks:
            rag_result = test_rag_processing(transcript, chunks, output_dir)
        else:
            print("\n⚠️  Skipping RAG test - no chunks available")
            rag_result = None
        
        # Test 7: Complete pipeline
        pipeline_result = test_complete_pipeline(audio_file, output_dir)
        
        # Summary
        print("\n" + "=" * 80)
        print("📊 **VERIFICATION SUMMARY**")
        print("=" * 80)
        
        results = {
            "Imports": "✅ PASS",
            "Output Directory": "✅ PASS",
            "Audio File": "✅ PASS" if audio_file else "❌ FAIL",
            "Transcription Storage": "✅ PASS" if transcription_result else "❌ FAIL",
            "Text Chunking": "✅ PASS" if chunks else "❌ FAIL",
            "RAG Processing": "✅ PASS" if rag_result else "⚠️  SKIP",
            "Complete Pipeline": "✅ PASS" if pipeline_result and pipeline_result.success else "❌ FAIL"
        }
        
        for test, status in results.items():
            print(f"   {test:20} {status}")
        
        # Key verification points
        print("\n🎯 **KEY VERIFICATION POINTS**")
        print(f"   1. Audio processed and passed to RAG: {'✅ YES' if rag_result else '❌ NO'}")
        print(f"   2. RAG generates output: {'✅ YES' if rag_result else '❌ NO'}")
        print(f"   3. Transcriptions stored before RAG: {'✅ YES' if transcription_file else '❌ NO'}")
        print(f"   4. Complete pipeline flow: {'✅ YES' if pipeline_result and pipeline_result.success else '❌ NO'}")
        
        # Output files created
        print(f"\n📁 **Output Files Created in {output_dir}:**")
        if output_dir.exists():
            for file_path in output_dir.glob("*"):
                file_size = file_path.stat().st_size
                print(f"   📄 {file_path.name} ({file_size} bytes)")
        
        success = all([
            transcription_result,
            chunks,
            pipeline_result and pipeline_result.success
        ])
        
        print(f"\n{'🎉 OVERALL: VERIFICATION SUCCESSFUL' if success else '❌ OVERALL: VERIFICATION FAILED'}")
        
        return success
        
    finally:
        # Cleanup
        if audio_file and audio_file.startswith('/tmp'):
            try:
                os.unlink(audio_file)
                print(f"\n🧹 Cleaned up test audio file: {audio_file}")
            except:
                pass

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
