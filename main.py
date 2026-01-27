"""
Inclusive Urban Simulation - Main Entry Point

This is the main entry point for the simulation system.
Run this file to test LLM connections or start the simulation.

Usage:
    python main.py              # Run LLM connection tests
    python main.py --help       # Show help (future)
"""

from typing import Dict
from llm_api import call_llm, get_enabled_models, get_disabled_models


# =============================================================================
# Test Functions
# =============================================================================

def test_llm_connection(model: str) -> bool:
    """
    Tests connection to a specific LLM model.

    Args:
        model: Model key to test

    Returns:
        True if connection successful, False otherwise
    """
    try:
        response = call_llm(
            model=model,
            memory=[],
            question="Say 'Hello' in one word."
        )
        print(f"    Response: {response.strip()[:50]}...")
        return True
    except Exception as e:
        print(f"    Error: {e}")
        return False


def test_enabled_models() -> Dict[str, bool]:
    """
    Tests connection to all enabled LLM models.

    Returns:
        Dictionary mapping model keys to test results (True/False)
    """
    results = {}
    enabled = get_enabled_models()
    disabled = get_disabled_models()

    print(f"Enabled models: {enabled}")
    print(f"Disabled models: {disabled}")
    print()

    for model in enabled:
        print(f"Testing {model}...")
        results[model] = test_llm_connection(model)
        status = "SUCCESS" if results[model] else "FAILED"
        print(f"  Result: {status}")
        print()

    return results


def test_memory_context() -> bool:
    """
    Tests that memory context is properly passed to the model.

    Returns:
        True if test passed, False otherwise
    """
    enabled = get_enabled_models()
    if not enabled:
        print("No models enabled for memory test")
        return False

    model = enabled[0]  # Use first enabled model
    print(f"Testing memory context with {model}...")

    try:
        # First call: introduce a fact
        memory = []
        response1 = call_llm(
            model=model,
            memory=memory,
            question="Remember this: The secret code is 'APPLE123'. Just say 'OK' to confirm."
        )
        print(f"  Step 1 response: {response1.strip()[:50]}...")

        # Add to memory
        memory.append({
            "question": "Remember this: The secret code is 'APPLE123'. Just say 'OK' to confirm.",
            "answer": response1
        })

        # Second call: recall the fact
        response2 = call_llm(
            model=model,
            memory=memory,
            question="What is the secret code I told you? Reply with just the code."
        )
        print(f"  Step 2 response: {response2.strip()[:50]}...")

        # Check if model remembered
        if "APPLE123" in response2.upper():
            print("  Memory test: PASSED")
            return True
        else:
            print("  Memory test: FAILED (code not found in response)")
            return False

    except Exception as e:
        print(f"  Memory test error: {e}")
        return False


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Inclusive Urban Simulation - LLM Interface Test")
    print("=" * 60)
    print()

    # Test enabled models
    results = test_enabled_models()

    # Print summary
    print("=" * 60)
    print("Connection Test Summary:")
    print("=" * 60)

    for model, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {model}: {status}")

    # Show disabled models
    disabled = get_disabled_models()
    if disabled:
        print(f"\n  Disabled (skipped): {', '.join(disabled)}")

    # Overall result
    if results:
        all_passed = all(results.values())
        print()
        print("=" * 60)
        print(f"Connection Tests: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
        print("=" * 60)

        # Run memory test if all connection tests passed
        if all_passed:
            print()
            print("=" * 60)
            print("Memory Context Test")
            print("=" * 60)
            memory_passed = test_memory_context()
            print()
            print("=" * 60)
            print(f"Memory Test: {'PASSED' if memory_passed else 'FAILED'}")
            print("=" * 60)
    else:
        print("\nNo models enabled for testing.")
