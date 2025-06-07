import pytest
from unittest.mock import patch, Mock, MagicMock
from intv import llm

@pytest.mark.llm
@pytest.mark.unit
def test_load_model_success():
    """Test successful model loading"""
    with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
        with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
            mock_tokenizer.return_value = Mock()
            mock_model.return_value = Mock()
            
            model, tokenizer = llm.load_model("test-model")
            
            assert model is not None
            assert tokenizer is not None
            mock_tokenizer.assert_called_once()
            mock_model.assert_called_once()

@pytest.mark.llm
@pytest.mark.unit
def test_load_model_failure():
    """Test model loading failure"""
    with patch('transformers.AutoTokenizer.from_pretrained', side_effect=Exception("Model not found")):
        with pytest.raises(Exception, match="Model not found"):
            llm.load_model("nonexistent-model")

@pytest.mark.llm
@pytest.mark.unit
def test_generate_text():
    """Test text generation"""
    mock_model = Mock()
    mock_tokenizer = Mock()
    
    # Mock tokenization
    mock_tokenizer.encode.return_value = [1, 2, 3, 4]
    mock_tokenizer.decode.return_value = "Generated response text"
    
    # Mock model generation
    mock_output = Mock()
    mock_output.sequences = [[1, 2, 3, 4, 5, 6, 7]]
    mock_model.generate.return_value = mock_output
    
    with patch.object(llm, 'model', mock_model):
        with patch.object(llm, 'tokenizer', mock_tokenizer):
            result = llm.generate_text("Test prompt", max_length=50)
            
            assert result == "Generated response text"
            mock_tokenizer.encode.assert_called_once()
            mock_model.generate.assert_called_once()
            mock_tokenizer.decode.assert_called_once()

@pytest.mark.llm
@pytest.mark.unit
def test_generate_text_with_parameters():
    """Test text generation with custom parameters"""
    mock_model = Mock()
    mock_tokenizer = Mock()
    
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_tokenizer.decode.return_value = "Custom response"
    
    mock_output = Mock()
    mock_output.sequences = [[1, 2, 3, 4, 5]]
    mock_model.generate.return_value = mock_output
    
    with patch.object(llm, 'model', mock_model):
        with patch.object(llm, 'tokenizer', mock_tokenizer):
            result = llm.generate_text(
                "Test prompt",
                max_length=100,
                temperature=0.8,
                top_p=0.9
            )
            
            # Check that generate was called with custom parameters
            call_args = mock_model.generate.call_args
            assert call_args[1]['max_length'] == 100
            assert call_args[1]['temperature'] == 0.8
            assert call_args[1]['top_p'] == 0.9

@pytest.mark.llm
@pytest.mark.unit
def test_encode_text():
    """Test text encoding"""
    mock_tokenizer = Mock()
    mock_tokenizer.encode.return_value = [101, 7592, 102]
    
    with patch.object(llm, 'tokenizer', mock_tokenizer):
        tokens = llm.encode_text("Hello world")
        
        assert tokens == [101, 7592, 102]
        mock_tokenizer.encode.assert_called_once_with("Hello world")

@pytest.mark.llm
@pytest.mark.unit
def test_decode_tokens():
    """Test token decoding"""
    mock_tokenizer = Mock()
    mock_tokenizer.decode.return_value = "Decoded text"
    
    with patch.object(llm, 'tokenizer', mock_tokenizer):
        text = llm.decode_tokens([101, 7592, 102])
        
        assert text == "Decoded text"
        mock_tokenizer.decode.assert_called_once_with([101, 7592, 102])

@pytest.mark.llm
@pytest.mark.slow
def test_batch_generate():
    """Test batch text generation"""
    mock_model = Mock()
    mock_tokenizer = Mock()
    
    # Mock batch tokenization
    mock_tokenizer.batch_encode_plus.return_value = {
        'input_ids': [[1, 2, 3], [4, 5, 6]],
        'attention_mask': [[1, 1, 1], [1, 1, 1]]
    }
    
    # Mock batch decoding
    mock_tokenizer.batch_decode.return_value = ["Response 1", "Response 2"]
    
    # Mock model generation
    mock_output = Mock()
    mock_output.sequences = [[1, 2, 3, 7, 8], [4, 5, 6, 9, 10]]
    mock_model.generate.return_value = mock_output
    
    with patch.object(llm, 'model', mock_model):
        with patch.object(llm, 'tokenizer', mock_tokenizer):
            prompts = ["Prompt 1", "Prompt 2"]
            results = llm.batch_generate(prompts)
            
            assert len(results) == 2
            assert results[0] == "Response 1"
            assert results[1] == "Response 2"

@pytest.mark.llm
@pytest.mark.unit
def test_calculate_perplexity():
    """Test perplexity calculation"""
    mock_model = Mock()
    mock_tokenizer = Mock()
    
    # Mock tokenization
    mock_tokenizer.encode.return_value = [1, 2, 3, 4]
    
    # Mock model forward pass
    mock_outputs = Mock()
    mock_outputs.logits = Mock()
    mock_outputs.loss = 2.5  # Cross-entropy loss
    mock_model.return_value = mock_outputs
    
    with patch.object(llm, 'model', mock_model):
        with patch.object(llm, 'tokenizer', mock_tokenizer):
            with patch('torch.exp') as mock_exp:
                mock_exp.return_value = 12.18  # exp(2.5)
                
                perplexity = llm.calculate_perplexity("Test text")
                
                assert perplexity == 12.18
                mock_exp.assert_called_once_with(2.5)

@pytest.mark.llm
@pytest.mark.unit
def test_get_model_info():
    """Test getting model information"""
    mock_model = Mock()
    mock_model.config.name_or_path = "test-model"
    mock_model.config.vocab_size = 50000
    mock_model.num_parameters.return_value = 1000000
    
    with patch.object(llm, 'model', mock_model):
        info = llm.get_model_info()
        
        assert info['name'] == "test-model"
        assert info['vocab_size'] == 50000
        assert info['num_parameters'] == 1000000
