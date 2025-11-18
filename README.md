# llm assisted epub reader

## Older karpathy's version of epub reader
![reader3](reader3.png)

## Updated epub reader with LLM assistant
![llm_reader](llm_reader.png)

A lightweight, self-hosted LLM assisted EPUB reader that lets you read through EPUB books one chapter at a time. This makes it very easy to read a chatper and know that there is an LLM reading along side you that nows the chatper and can answer your queries rike a pair-reader. Basically - get epub books (e.g. [Project Gutenberg](https://www.gutenberg.org/) has many), open them up in this reader, copy paste text around to your favorite LLM, and read together and along.

This project was forked from [Karpathy's reader3 repo](https://github.com/karpathy/reader3?tab=readme-ov-file) and modifications have been made on top of it to support LLM assisted reading

## Usage

The project uses [uv](https://docs.astral.sh/uv/). So for example, download [Dracula EPUB3](https://www.gutenberg.org/ebooks/345) to this directory as `dracula.epub`, then:

```bash
uv run reader3.py dracula.epub
```

This creates the directory `dracula_data`, which registers the book to your local library. We can then run the server:

```bash
uv run server.py
```

And visit [localhost:8123](http://localhost:8123/) to see your current Library. You can easily add more books, or delete them from your library by deleting the folder. It's not supposed to be complicated or complex.

