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
async def redirect_to_first_chapter(request: Request, book_id: str, format: str = Query("html", description="Display format: 'html' or 'text'")):
    """Helper to just go to chapter 0."""
    return await read_chapter(request=request, book_id=book_id, chapter_index=0, format=format)

@app.get("/read/{book_id}/{chapter_index}", response_class=HTMLResponse)
async def read_chapter(
    request: Request, 
    book_id: str, 
    chapter_index: int,
    format: str = Query("html", description="Display format: 'html' or 'text'")
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

    return templates.TemplateResponse("reader.html", {
        "request": request,
        "book": book,
        "current_chapter": current_chapter,
        "chapter_index": chapter_index,
        "book_id": book_id,
        "prev_idx": prev_idx,
        "next_idx": next_idx,
        "format": format,
        "chapter_title_display": chapter_title_display
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
        
        # Check if the response itself is an error message (starts with "Error:")
        if response.startswith("Error:"):
            # Extract the error message
            error_msg = response.replace("Error:", "").strip()
            # Check if it contains HTML
            if "<!DOCTYPE html>" in error_msg or "<html" in error_msg.lower():
                error_msg = "API service temporarily unavailable. Please try again in a few moments."
            return JSONResponse(
                {"error": error_msg},
                status_code=500
            )
        
        return JSONResponse({"response": response})
    except Exception as e:
        error_msg = str(e)
        # If error message is HTML (like Cloudflare error page), provide a cleaner message
        if "<!DOCTYPE html>" in error_msg or "<html" in error_msg.lower():
            error_msg = "API service temporarily unavailable. Please try again in a few moments."
        elif len(error_msg) > 200:
            error_msg = error_msg[:200] + "..."
        return JSONResponse(
            {"error": f"Error calling LLM: {error_msg}"},
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
