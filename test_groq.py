#!/usr/bin/env python3
"""
Test script to check if Groq API is working
"""

import sys
from groq import Groq

def test_groq_api(api_key: str):
    """Test Groq API with a simple request"""
    print("Testing Groq API...")
    print(f"API Key provided: {'Yes' if api_key else 'No'}")
    print(f"API Key length: {len(api_key) if api_key else 0}")
    print("-" * 50)
    
    if not api_key:
        print("ERROR: No API key provided")
        print("Usage: python test_groq.py <your_groq_api_key>")
        return False
    
    try:
        # Initialize client
        print("1. Initializing Groq client...")
        client = Groq(api_key=api_key)
        print("   ✓ Client initialized successfully")
        
        # Make a simple test call
        print("\n2. Making test API call...")
        print("   Model: llama-3.1-8b-instant")
        print("   Message: 'Say hello'")
        
        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": "Say hello"}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=50
        )
        
        print("   ✓ API call successful!")
        print(f"\n3. Response received:")
        print(f"   Content: {response.choices[0].message.content}")
        print(f"   Model used: {response.model}")
        print(f"   Finish reason: {response.choices[0].finish_reason}")
        
        # Try the primary model too
        print("\n4. Testing primary model (llama-3.3-70b-versatile)...")
        try:
            response2 = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": "Say hello"}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                max_tokens=50
            )
            print("   ✓ Primary model working!")
            print(f"   Content: {response2.choices[0].message.content}")
        except Exception as e:
            print(f"   ✗ Primary model failed: {str(e)}")
            print("   (This is okay, fallback model works)")
        
        print("\n" + "=" * 50)
        print("RESULT: Groq API is working correctly!")
        return True
        
    except Exception as e:
        error_str = str(e)
        print(f"\n✗ ERROR occurred:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {error_str[:500]}")
        
        # Check for HTML errors
        if "<!DOCTYPE html>" in error_str or "<html" in error_str.lower():
            print("\n   ⚠️  Detected HTML error page (likely Cloudflare 500 error)")
            print("   This suggests Groq's API service is temporarily down")
        elif "500" in error_str or "Internal Server Error" in error_str:
            print("\n   ⚠️  Detected 500 Internal Server Error")
            print("   Groq's API service may be experiencing issues")
        elif "401" in error_str or "Unauthorized" in error_str:
            print("\n   ⚠️  Authentication error - check your API key")
        elif "429" in error_str or "rate limit" in error_str.lower():
            print("\n   ⚠️  Rate limit exceeded - try again later")
        
        print("\n" + "=" * 50)
        print("RESULT: Groq API test FAILED")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_groq.py <your_groq_api_key>")
        print("\nOr set GROQ_API_KEY environment variable:")
        print("  export GROQ_API_KEY='your_key_here'")
        print("  python test_groq.py")
        sys.exit(1)
    
    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Try to get from environment if not provided
    import os
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")
    
    success = test_groq_api(api_key)
    sys.exit(0 if success else 1)

