import pytest
from model_registry import ModelRegistry
from model_registry.exceptions import StoreError


@pytest.mark.e2e
def test_verify_hf_functionality(client: ModelRegistry):
    """Simple test to verify HF functionality is working - based on existing test pattern."""
    # Skip if huggingface-hub is not available
    pytest.importorskip("huggingface_hub")
    
    # Test parameters (using same repo as existing tests)
    name = "openai-community/gpt2"
    version = "verify-1.0.0"
    author = "verification-author"

    # Register HF model
    registered_model = client.register_hf_model(
        name,
        "onnx/decoder_model.onnx",
        author=author,
        version=version,
        model_format_name="onnx",
        model_format_version="1.0",
    )
    
    # Basic assertions
    assert registered_model
    assert registered_model.id
    
    # Verify model version was created correctly
    mv = client.get_model_version(name, version)
    assert mv
    assert mv.author == author
    assert mv.custom_properties
    
    # Verify HF-specific metadata
    assert mv.custom_properties["model_author"] == author
    assert mv.custom_properties["model_origin"] == "huggingface_hub"
    assert mv.custom_properties["repo"] == name
    assert "huggingface.co" in mv.custom_properties["source_uri"]
    
    # Verify model artifact exists
    assert client.get_model_artifact(name, version)
    
    print("✅ HF functionality verification passed!")


def test_hf_import_error_handling():
    """Test that proper error is raised when huggingface-hub is not available."""
    # This test simulates missing dependency
    import sys
    
    # Create a mock registry (won't actually connect since we're testing error handling)
    try:
        mock_registry = ModelRegistry("http://localhost", 8080, author="test", is_secure=False)
        
        # Temporarily hide huggingface_hub
        hf_backup = sys.modules.get('huggingface_hub')
        if 'huggingface_hub' in sys.modules:
            del sys.modules['huggingface_hub']
        
        # This should raise StoreError about missing dependency
        with pytest.raises(StoreError) as exc_info:
            mock_registry.register_hf_model(
                "test/repo",
                "model.onnx",
                version="1.0.0",
                model_format_name="onnx",
                model_format_version="1",
            )
        
        # Verify error message
        assert "huggingface-hub" in str(exc_info.value).lower()
        print("✅ Error handling verification passed!")
        
    finally:
        # Restore if it was there
        if hf_backup:
            sys.modules['huggingface_hub'] = hf_backup


def test_hf_method_exists():
    """Quick test to verify the register_hf_model method exists."""
    # This doesn't require a running server, just checks the method exists
    mock_registry = ModelRegistry("http://mock", 8080, author="test", is_secure=False)
    
    # Check method exists
    assert hasattr(mock_registry, 'register_hf_model')
    assert callable(getattr(mock_registry, 'register_hf_model'))
    
    print("✅ HF method existence verification passed!")


if __name__ == "__main__":
    # Run standalone tests (ones that don't need server)
    print("Running HF verification tests...")
    
    test_hf_method_exists()
    test_hf_import_error_handling()
    
    print("\n✅ Standalone HF tests passed!")
    print("\nTo run the full e2e test with a model registry server:")
    print("pytest simple_hf_test.py::test_verify_hf_functionality --e2e") 