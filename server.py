import os
import pickle
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse, PlainTextResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from reader3 import Book, BookMetadata, ChapterContent, TOCEntry
from llm_service import GroqLLMService


class ChatRequest(BaseModel):
    api_key: str
    message: str

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Where are the book folders located?
BOOKS_DIR = "."

@lru_cache(maxsize=10)
def load_book_cached(folder_name: str) -> Optional[Book]:
    """
    Loads the book from the pickle file.
    Cached so we don't re-read the disk on every click.
    """
    file_path = os.path.join(BOOKS_DIR, folder_name, "book.pkl")
    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, "rb") as f:
            book = pickle.load(f)
        return book
    except Exception as e:
        print(f"Error loading book {folder_name}: {e}")
        return None


def get_chapter_title_from_toc(book: Book, chapter_href: str) -> str:
    """
    Find the actual chapter title from TOC by matching the href.
    Recursively searches through the TOC structure.
    """
    def search_toc_recursive(toc_entries):
        for entry in toc_entries:
            # Check if this TOC entry matches the chapter href
            if entry.file_href == chapter_href or entry.href.split('#')[0] == chapter_href:
                return entry.title
            # Recursively search children
            if entry.children:
                result = search_toc_recursive(entry.children)
                if result:
                    return result
        return None
    
    title = search_toc_recursive(book.toc)
    return title if title else None


def get_llmstxt_path(book_id: str, chapter_index: int) -> str:
    """Get the file path for storing llms.txt for a chapter."""
    safe_book_id = os.path.basename(book_id)
    llmstxt_dir = os.path.join(BOOKS_DIR, safe_book_id, "llmstxt")
    os.makedirs(llmstxt_dir, exist_ok=True)
    return os.path.join(llmstxt_dir, f"chapter_{chapter_index}.txt")


def load_llmstxt(book_id: str, chapter_index: int) -> Optional[str]:
    """Load llms.txt content from file if it exists."""
    llmstxt_path = get_llmstxt_path(book_id, chapter_index)
    if os.path.exists(llmstxt_path):
        try:
            with open(llmstxt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error loading llms.txt: {e}")
            return None
    return None


def save_llmstxt(book_id: str, chapter_index: int, content: str) -> bool:
    """Save llms.txt content to file."""
    try:
        llmstxt_path = get_llmstxt_path(book_id, chapter_index)
        with open(llmstxt_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Error saving llms.txt: {e}")
        return False

@app.get("/", response_class=HTMLResponse)
async def library_view(request: Request):
    """Lists all available processed books."""
    books = []

    # Scan directory for folders ending in '_data' that have a book.pkl
    if os.path.exists(BOOKS_DIR):
        for item in os.listdir(BOOKS_DIR):
            if item.endswith("_data") and os.path.isdir(item):
                # Try to load it to get the title
                book = load_book_cached(item)
                if book:
                    books.append({
                        "id": item,
                        "title": book.metadata.title,
                        "author": ", ".join(book.metadata.authors),
                        "chapters": len(book.spine)
                    })

    return templates.TemplateResponse("library.html", {"request": request, "books": books})

@app.get("/read/{book_id}", response_class=HTMLResponse)
async def redirect_to_first_chapter(request: Request, book_id: str, format: str = Query("html", description="Display format: 'html', 'text', or 'llmstxt'")):
    """Helper to just go to chapter 0."""
    return await read_chapter(request=request, book_id=book_id, chapter_index=0, format=format)

@app.get("/read/{book_id}/{chapter_index}", response_class=HTMLResponse)
async def read_chapter(
    request: Request, 
    book_id: str, 
    chapter_index: int,
    format: str = Query("html", description="Display format: 'html', 'text', or 'llmstxt'")
):
    """The main reader interface."""
    book = load_book_cached(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    if chapter_index < 0 or chapter_index >= len(book.spine):
        raise HTTPException(status_code=404, detail="Chapter not found")

    current_chapter = book.spine[chapter_index]

    # Calculate Prev/Next links
    prev_idx = chapter_index - 1 if chapter_index > 0 else None
    next_idx = chapter_index + 1 if chapter_index < len(book.spine) - 1 else None

    # Get actual chapter title from TOC for display
    chapter_title_display = get_chapter_title_from_toc(book, current_chapter.href)
    if not chapter_title_display:
        chapter_title_display = current_chapter.title

    # Check if llms.txt exists for this chapter
    llmstxt_content = load_llmstxt(book_id, chapter_index)
    has_llmstxt = llmstxt_content is not None

    return templates.TemplateResponse("reader.html", {
        "request": request,
        "book": book,
        "current_chapter": current_chapter,
        "chapter_index": chapter_index,
        "book_id": book_id,
        "prev_idx": prev_idx,
        "next_idx": next_idx,
        "format": format,
        "chapter_title_display": chapter_title_display,
        "has_llmstxt": has_llmstxt,
        "llmstxt_content": llmstxt_content
    })

@app.get("/api/read/{book_id}/{chapter_index}/text", response_class=PlainTextResponse)
async def get_chapter_text(book_id: str, chapter_index: int):
    """
    API endpoint to get plain text of a chapter.
    Returns just the plain text without HTML wrapper.
    """
    book = load_book_cached(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    if chapter_index < 0 or chapter_index >= len(book.spine):
        raise HTTPException(status_code=404, detail="Chapter not found")

    current_chapter = book.spine[chapter_index]
    return PlainTextResponse(current_chapter.text)

@app.get("/api/read/{book_id}/{chapter_index}/llmstxt", response_class=PlainTextResponse)
async def get_chapter_llmstxt(book_id: str, chapter_index: int):
    """
    API endpoint to get llms.txt format of a chapter.
    Returns just the llms.txt content without HTML wrapper.
    """
    book = load_book_cached(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    if chapter_index < 0 or chapter_index >= len(book.spine):
        raise HTTPException(status_code=404, detail="Chapter not found")

    llmstxt_content = load_llmstxt(book_id, chapter_index)
    if llmstxt_content is None:
        raise HTTPException(
            status_code=404, 
            detail="llms.txt format not available for this chapter. Generate it using the 'Generate llms.txt' button."
        )
    
    return PlainTextResponse(llmstxt_content)

@app.get("/api/check-llmstxt/{book_id}/{chapter_index}")
async def check_llmstxt(book_id: str, chapter_index: int):
    """Check if llms.txt exists for a chapter."""
    book = load_book_cached(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    if chapter_index < 0 or chapter_index >= len(book.spine):
        raise HTTPException(status_code=404, detail="Chapter not found")

    has_llmstxt = load_llmstxt(book_id, chapter_index) is not None
    return JSONResponse({"has_llmstxt": has_llmstxt})


class GenerateLlmstxtRequest(BaseModel):
    api_key: str


@app.post("/api/generate-llmstxt/{book_id}/{chapter_index}")
async def generate_llmstxt(book_id: str, chapter_index: int, request_data: GenerateLlmstxtRequest):
    """
    Generate llms.txt format for a chapter on-demand.
    """
    book = load_book_cached(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    if chapter_index < 0 or chapter_index >= len(book.spine):
        raise HTTPException(status_code=404, detail="Chapter not found")

    api_key = request_data.api_key.strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="API key is required")

    current_chapter = book.spine[chapter_index]
    
    # Get chapter title
    chapter_title = get_chapter_title_from_toc(book, current_chapter.href)
    if not chapter_title:
        chapter_title = current_chapter.title

    try:
        llm_service = GroqLLMService(api_key=api_key)
        llmstxt_content = llm_service.generate_llmstxt_from_html(
            html_content=current_chapter.content,
            chapter_title=chapter_title
        )
        
        if llmstxt_content is None:
            return JSONResponse(
                {"error": "Failed to generate llms.txt. Please check your API key and try again."},
                status_code=500
            )
        
        # Save to file
        if save_llmstxt(book_id, chapter_index, llmstxt_content):
            return JSONResponse({
                "success": True,
                "message": "llms.txt generated successfully"
            })
        else:
            return JSONResponse(
                {"error": "Failed to save llms.txt file."},
                status_code=500
            )
            
    except Exception as e:
        error_msg = str(e)
        if "<!DOCTYPE html>" in error_msg or "<html" in error_msg.lower():
            error_msg = "API service temporarily unavailable. Please try again in a few moments."
        elif len(error_msg) > 200:
            error_msg = error_msg[:200] + "..."
        return JSONResponse(
            {"error": f"Error generating llms.txt: {error_msg}"},
            status_code=500
        )

@app.post("/api/chat/{book_id}/{chapter_index}")
async def chat_with_chapter(book_id: str, chapter_index: int, request_data: ChatRequest):
    """
    Chat endpoint for LLM interaction with chapter context.
    """
    book = load_book_cached(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    if chapter_index < 0 or chapter_index >= len(book.spine):
        raise HTTPException(status_code=404, detail="Chapter not found")

    api_key = request_data.api_key.strip()
    user_message = request_data.message.strip()

    if not api_key:
        raise HTTPException(status_code=400, detail="API key is required")
    
    if not user_message:
        raise HTTPException(status_code=400, detail="Message is required")

    current_chapter = book.spine[chapter_index]
    
    # Prefer llms.txt if available, otherwise use plain text
    llmstxt_content = load_llmstxt(book_id, chapter_index)
    if llmstxt_content:
        chapter_text = llmstxt_content
    else:
        chapter_text = current_chapter.text
    
    # Get actual chapter title from TOC, fallback to spine title
    chapter_title = get_chapter_title_from_toc(book, current_chapter.href)
    if not chapter_title:
        chapter_title = current_chapter.title

    try:
        llm_service = GroqLLMService(api_key=api_key)
        response = llm_service.chat_with_chapter(
            chapter_text=chapter_text,
            user_message=user_message,
            chapter_title=chapter_title
        )
        
        return JSONResponse({"response": response})
    except Exception as e:
        error_msg = str(e)
        # The error message from llm_service should already be user-friendly
        # Just clean it up if it contains HTML
        if "<!DOCTYPE html>" in error_msg or "<html" in error_msg.lower():
            error_msg = "API service temporarily unavailable. Please try again in a few moments."
        elif len(error_msg) > 300:
            # Truncate very long error messages but keep them informative
            error_msg = error_msg[:300] + "..."
        
        # Return the error message directly (it's already formatted by llm_service)
        return JSONResponse(
            {"error": error_msg},
            status_code=500
        )

@app.get("/read/{book_id}/images/{image_name}")
async def serve_image(book_id: str, image_name: str):
    """
    Serves images specifically for a book.
    The HTML contains <img src="images/pic.jpg">.
    The browser resolves this to /read/{book_id}/images/pic.jpg.
    """
    # Security check: ensure book_id is clean
    safe_book_id = os.path.basename(book_id)
    safe_image_name = os.path.basename(image_name)

    img_path = os.path.join(BOOKS_DIR, safe_book_id, "images", safe_image_name)

    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(img_path)

if __name__ == "__main__":
    import uvicorn
    print("Starting server at http://127.0.0.1:8123")
    uvicorn.run(app, host="127.0.0.1", port=8123)
