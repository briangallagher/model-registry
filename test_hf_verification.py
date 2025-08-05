import os
import pytest
from model_registry import ModelRegistry
from model_registry.exceptions import StoreError


def test_hf_dependency_available():
    """Test that huggingface-hub dependency is available for import."""
    try:
        import huggingface_hub
        assert huggingface_hub is not None
        print("✓ huggingface-hub is available")
    except ImportError:
        pytest.fail("huggingface-hub is not installed. Install with: pip install model-registry[hf]")


def test_hf_functionality_without_dependency():
    """Test that register_hf_model raises appropriate error when huggingface-hub is not available."""
    # Mock the import to simulate missing dependency
    import sys
    import types
    
    # Create a mock ModelRegistry for this test
    # Note: In real usage, you'd use the client fixture from conftest.py
    mock_registry = ModelRegistry("http://localhost", 8080, author="test", is_secure=False)
    
    # Temporarily remove huggingface_hub from sys.modules if it exists
    hf_hub_backup = sys.modules.get('huggingface_hub')
    if 'huggingface_hub' in sys.modules:
        del sys.modules['huggingface_hub']
    
    try:
        # This should raise a StoreError about missing huggingface-hub
        with pytest.raises(StoreError) as exc_info:
            mock_registry.register_hf_model(
                "openai-community/gpt2",
                "onnx/decoder_model.onnx",
                version="1.0.0",
                model_format_name="onnx",
                model_format_version="1",
            )
        
        # Verify the error message mentions huggingface-hub
        error_msg = str(exc_info.value).lower()
        assert "huggingface-hub" in error_msg
        assert "not installed" in error_msg
        print("✓ Proper error handling when huggingface-hub is missing")
        
    finally:
        # Restore huggingface_hub if it was there originally
        if hf_hub_backup:
            sys.modules['huggingface_hub'] = hf_hub_backup


@pytest.mark.e2e
def test_hf_import_verification(client: ModelRegistry):
    """Test HF model import functionality (requires actual model registry server)."""
    # Skip if huggingface-hub is not available
    pytest.importorskip("huggingface_hub")
    
    # Test parameters
    repo_name = "openai-community/gpt2"
    model_path = "onnx/decoder_model.onnx"
    version = "test-1.0.0"
    author = "test-hf-author"
    
    print(f"Testing HF import for {repo_name}")
    
    try:
        # Register the HF model
        registered_model = client.register_hf_model(
            repo_name,
            model_path,
            author=author,
            version=version,
            model_format_name="onnx",
            model_format_version="1.0",
            description="Test HF model import"
        )
        
        assert registered_model is not None
        assert registered_model.id is not None
        print(f"✓ Successfully registered HF model with ID: {registered_model.id}")
        
        # Verify the model version was created with correct metadata
        model_version = client.get_model_version(repo_name, version)
        assert model_version is not None
        assert model_version.author == author
        
        # Verify HF-specific custom properties
        assert model_version.custom_properties is not None
        custom_props = model_version.custom_properties
        
        # Check for HF-specific metadata
        assert custom_props.get("model_author") == author
        assert custom_props.get("model_origin") == "huggingface_hub"
        assert custom_props.get("repo") == repo_name
        
        # Verify source URI is correctly formatted
        source_uri = custom_props.get("source_uri")
        assert source_uri is not None
        assert "huggingface.co" in source_uri
        assert repo_name in source_uri
        assert model_path in source_uri
        
        print(f"✓ Model version created with correct HF metadata")
        print(f"  - Source URI: {source_uri}")
        print(f"  - Model origin: {custom_props.get('model_origin')}")
        
        # Verify model artifact was created
        model_artifact = client.get_model_artifact(repo_name, version)
        assert model_artifact is not None
        print(f"✓ Model artifact created successfully")
        
        return True
        
    except Exception as e:
        pytest.fail(f"HF import test failed: {str(e)}")


def test_hf_import_with_custom_git_ref():
    """Test HF import with custom git reference (unit test style)."""
    pytest.importorskip("huggingface_hub")
    
    # This is a unit test that doesn't require a running server
    # We're just testing the method exists and accepts the git_ref parameter
    
    # Create a mock registry (this won't actually connect)
    mock_registry = ModelRegistry("http://mock", 8080, author="test", is_secure=False)
    
    # Test that the method exists and accepts git_ref parameter
    assert hasattr(mock_registry, 'register_hf_model')
    
    # Get the method signature to verify git_ref parameter
    import inspect
    sig = inspect.signature(mock_registry.register_hf_model)
    
    assert 'git_ref' in sig.parameters
    assert sig.parameters['git_ref'].default == 'main'
    
    print("✓ register_hf_model method has git_ref parameter with default 'main'")


if __name__ == "__main__":
    print("Running HF verification tests...")
    
    # Run individual tests
    test_hf_dependency_available()
    test_hf_functionality_without_dependency()
    test_hf_import_with_custom_git_ref()
    
    print("\n🎉 All HF verification tests passed!")
    print("\nTo run the full e2e test, use:")
    print("pytest test_hf_verification.py::test_hf_import_verification --e2e") 