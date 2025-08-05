#!/usr/bin/env python3

def test_hf_extra():
    """Test HF (Hugging Face) extra availability."""
    try:
        from huggingface_hub import HfApi  # main class used by HF functionality
        print("ModelRegistry HF client:", HfApi)
    except ImportError as e:
        print("ModelRegistry HF submodule not available:", e)


def test_olot_extra():
    """Test olot extra availability."""
    try:
        from olot.backend.skopeo import skopeo_pull  # adjust class name as appropriate
        print("ModelRegistry olot client:", skopeo_pull)
    except ImportError as e:
        print("ModelRegistry olot submodule not available:", e)


def test_olot_oras_backend():
    """Test olot oras backend availability."""
    try:
        from olot.backend.oras_cp import oras_pull  # adjust class name as appropriate
        print("ModelRegistry olot oras client:", oras_pull)
    except ImportError as e:
        print("ModelRegistry olot oras submodule not available:", e)


def test_olot_basics():
    """Test olot basics functionality."""
    try:
        from olot.basics import oci_layers_on_top  # main function used by olot functionality
        print("ModelRegistry olot basics:", oci_layers_on_top)
    except ImportError as e:
        print("ModelRegistry olot basics submodule not available:", e)


if __name__ == "__main__":
    print("Testing ModelRegistry extras availability...\n")
    
    print("1. Testing HF extra:")
    test_hf_extra()
    
    print("\n2. Testing olot extra (skopeo backend):")
    test_olot_extra()
    
    print("\n3. Testing olot extra (oras backend):")
    test_olot_oras_backend()
    
    print("\n4. Testing olot basics:")
    test_olot_basics() 