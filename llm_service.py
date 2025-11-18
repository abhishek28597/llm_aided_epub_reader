"""
LLM Service for Groq Integration
"""

from typing import List, Dict
from groq import Groq


class GroqLLMService:
    def __init__(self, api_key: str):
        """
        Initialize Groq client with API key
        
        Args:
            api_key: Groq API key
        """
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
        self.fallback_model = "llama-3.1-8b-instant"
    
    def _make_api_call(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 1000):
        """
        Make API call with automatic fallback on any error
        
        Args:
            messages: List of message dictionaries
            temperature: Temperature setting
            max_tokens: Maximum tokens
            
        Returns:
            Response from the API
            
        Raises:
            Exception: If both primary and fallback models fail
        """
        last_error = None
        try:
            return self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens
            )
        except Exception as e:
            last_error = e
            error_str = str(e)
            # If we get an HTML error page, don't try fallback - it will likely fail too
            if "<!DOCTYPE html>" in error_str or "<html" in error_str.lower() or "500" in error_str:
                raise Exception("API service temporarily unavailable. Please try again in a few moments.")
            # Try fallback model on other errors
            try:
                return self.client.chat.completions.create(
                    messages=messages,
                    model=self.fallback_model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            except Exception as e2:
                last_error = e2
                raise last_error
    
    def chat_with_chapter(self, chapter_text: str, user_message: str, chapter_title: str = "") -> str:
        """
        Chat about a specific chapter with context
        
        Args:
            chapter_text: The plain text content of the chapter
            user_message: User's question or message
            chapter_title: Optional chapter title for context
            
        Returns:
            LLM response string
        """
        # Estimate token count (rough: 1 token ≈ 4 characters)
        # Groq models typically have 128k context, but we'll be conservative
        # Reserve space for system prompt, user message, and response
        max_chapter_length = 100000  # ~25k tokens for chapter, leaving room for conversation
        
        if len(chapter_text) > max_chapter_length:
            chapter_text = chapter_text[:max_chapter_length] + "\n\n[Chapter content truncated due to length...]"
        
        # Build context
        context = f"You are a helpful reading assistant. The user is reading a book chapter and has a question about it.\n\n"
        if chapter_title:
            context += f"Current Chapter: {chapter_title}\n\n"
        context += f"Chapter Content:\n{chapter_text}\n\n"
        context += "Answer the user's question based on the chapter content above. Be concise and helpful."
        
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": user_message}
        ]
        
        try:
            response = self._make_api_call(
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            error_str = str(e)
            # Check if error contains HTML (like Cloudflare error pages)
            if "<!DOCTYPE html>" in error_str or "<html" in error_str.lower():
                return "API service temporarily unavailable. Please try again in a few moments."
            # Check for common Groq API errors
            if "500" in error_str or "Internal Server Error" in error_str:
                return "The API service is experiencing issues. Please try again in a few moments."
            # Return a clean error message
            if len(error_str) > 200:
                # Truncate very long error messages
                error_str = error_str[:200] + "..."
            return f"Error: {error_str}"

