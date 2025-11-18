"""
LLM Service for Groq Integration
"""

from typing import List, Dict, Optional
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
            error_str = str(e).lower()
            original_error = str(e)
            
            # Log the original error for debugging (can be removed in production)
            print(f"Groq API Error: {original_error}")
            
            # Check for specific error types
            if "authentication" in error_str or ("invalid" in error_str and "key" in error_str):
                raise Exception("Invalid API key. Please check your Groq API key and try again.")
            
            if "rate limit" in error_str or "429" in str(e):
                raise Exception("Rate limit exceeded. Please wait a moment and try again.")
            
            if "context length" in error_str or ("token" in error_str and ("limit" in error_str or "exceeded" in error_str)):
                raise Exception("Content too long. The chapter content exceeds the model's token limit. Try using plain text format instead of llms.txt.")
            
            # If we get an HTML error page or HTTP 500, don't try fallback
            if "<!DOCTYPE html>" in str(e) or "<html" in error_str or ("500" in str(e) and "internal server error" in error_str):
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
                # Re-check error types for fallback
                error_str2 = str(e2).lower()
                if "authentication" in error_str2 or "invalid" in error_str2 and "key" in error_str2:
                    raise Exception("Invalid API key. Please check your Groq API key and try again.")
                if "rate limit" in error_str2 or "429" in str(e2):
                    raise Exception("Rate limit exceeded. Please wait a moment and try again.")
                if "context length" in error_str2 or "token" in error_str2 and ("limit" in error_str2 or "exceeded" in error_str2):
                    raise Exception("Content too long. The chapter content exceeds the model's token limit. Try using plain text format instead of llms.txt.")
                raise last_error
    
    def chat_with_chapter(self, chapter_text: str, user_message: str, chapter_title: str = "") -> str:
        """
        Chat about a specific chapter with context
        
        Args:
            chapter_text: The plain text or llms.txt content of the chapter
            user_message: User's question or message
            chapter_title: Optional chapter title for context
            
        Returns:
            LLM response string
            
        Raises:
            Exception: If API call fails, raises with descriptive error message
        """
        # Estimate token count (rough: 1 token ≈ 4 characters)
        # Groq models typically have 128k context, but we'll be conservative
        # Reserve space for system prompt, user message, and response
        # For llms.txt, content might be longer, so we allow more
        max_chapter_length = 200000  # ~50k tokens for chapter, leaving room for conversation
        
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
        
        # Let exceptions propagate - they will have descriptive messages from _make_api_call
        response = self._make_api_call(
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    
    def generate_llmstxt_from_html(self, html_content: str, chapter_title: str = "") -> Optional[str]:
        """
        Convert HTML content to llms.txt format (structured markdown).
        
        Args:
            html_content: The HTML content of the chapter
            chapter_title: Optional chapter title for context
            
        Returns:
            llms.txt formatted markdown string, or None if generation fails
        """
        # Truncate HTML if too long (estimate: 1 token ≈ 4 characters)
        # Reserve space for prompt and response
        max_html_length = 200000  # ~50k tokens for HTML, leaving room for prompt and response
        
        if len(html_content) > max_html_length:
            html_content = html_content[:max_html_length] + "\n\n[Content truncated due to length...]"
        
        # Build prompt for conversion following llms.txt specification (https://llmstxt.org/)
        system_prompt = """You are a content conversion assistant. Convert the provided HTML content into llms.txt format following the official specification at llmstxt.org.

The llms.txt format must follow this exact structure (in order):

1. **H1 heading** (required): The chapter title as a single H1 heading (# Title)

2. **Blockquote summary** (required): A brief summary of the chapter content in a blockquote (> summary text). This should contain key information necessary for understanding the chapter.

3. **Content sections** (optional): The main chapter content as markdown (paragraphs, lists, emphasis, code blocks, etc.). For book chapters, you may use H2 headings to organize major subsections within the content if the original HTML has clear section breaks. Otherwise, use paragraphs and lists without headings.

4. **H2 sections with file lists** (optional): If the chapter references external resources, links, or citations, create H2 sections followed by markdown lists with hyperlinks in the format: - [Link title](url): Optional description. If there are no external references, omit this section.

Important formatting rules:
- Start with exactly one H1 heading with the chapter title
- Follow with exactly one blockquote containing a concise 1-3 sentence summary
- Convert all HTML to clean markdown (paragraphs, lists, emphasis, code blocks, etc.)
- Preserve the semantic structure and meaning of the original content
- Remove all HTML tags and convert to appropriate markdown syntax
- Use H2 headings for major subsections within the chapter content (if present in original)
- Use H2 headings for file list sections (if there are external references)
- Maintain readability and logical flow
- Keep the summary blockquote concise and informative

Return only the llms.txt formatted markdown, no additional commentary."""
        
        user_prompt = f"Convert the following HTML chapter content to llms.txt format:\n\n{html_content}"
        
        if chapter_title:
            user_prompt = f"Chapter Title: {chapter_title}\n\n{user_prompt}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # Use higher max_tokens for markdown generation (longer output expected)
            response = self._make_api_call(
                messages=messages,
                temperature=0.3,  # Lower temperature for more consistent formatting
                max_tokens=8000   # Allow longer responses for full chapter conversion
            )
            return response.choices[0].message.content
        except Exception as e:
            # Log error but don't raise - return None so processing can continue
            error_str = str(e)
            # Check if error contains HTML (like Cloudflare error pages)
            if "<!DOCTYPE html>" in error_str or "<html" in error_str.lower():
                print(f"Warning: API service unavailable for llms.txt generation")
            else:
                print(f"Warning: Failed to generate llms.txt: {error_str[:200]}")
            return None

