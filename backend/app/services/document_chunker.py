"""
Document Chunking Service for RAG

Processes textbook markdown files into semantic chunks for embedding and retrieval.
"""

import os
import re
from typing import List, Dict, Any
from pathlib import Path


class DocumentChunk:
    """Represents a chunk of document content with metadata."""

    def __init__(
        self,
        content: str,
        module: str,
        chapter: str,
        section: str,
        file_path: str,
        chunk_index: int
    ):
        self.content = content
        self.module = module
        self.chapter = chapter
        self.section = section
        self.file_path = file_path
        self.chunk_index = chunk_index

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for storage."""
        return {
            "content": self.content,
            "metadata": {
                "module": self.module,
                "chapter": self.chapter,
                "section": self.section,
                "file_path": self.file_path,
                "chunk_index": self.chunk_index
            }
        }


class DocumentChunker:
    """Chunks textbook content into semantic sections."""

    def __init__(self, docs_path: str, max_chunk_size: int = 1000):
        """
        Initialize document chunker.

        Args:
            docs_path: Path to docs directory containing markdown files
            max_chunk_size: Maximum characters per chunk
        """
        self.docs_path = Path(docs_path)
        self.max_chunk_size = max_chunk_size

    def chunk_documents(self) -> List[DocumentChunk]:
        """
        Chunk all markdown documents in the textbook.

        Returns:
            List of DocumentChunk objects
        """
        chunks = []

        # Find all markdown files in module directories
        for module_dir in self.docs_path.glob("module*"):
            if not module_dir.is_dir():
                continue

            module_name = module_dir.name

            for md_file in module_dir.glob("*.md"):
                if md_file.name == "intro.md":
                    continue  # Skip intro files for now

                file_chunks = self._chunk_file(
                    md_file,
                    module_name
                )
                chunks.extend(file_chunks)

        return chunks

    def _chunk_file(
        self,
        file_path: Path,
        module_name: str
    ) -> List[DocumentChunk]:
        """
        Chunk a single markdown file.

        Args:
            file_path: Path to markdown file
            module_name: Name of the module (e.g., module1-ros2)

        Returns:
            List of chunks from this file
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract chapter name from filename
        chapter_name = file_path.stem.replace('chapter', 'Chapter ')

        # Split content by h2 headers (##)
        sections = self._split_by_headers(content)

        chunks = []
        for section_title, section_content in sections:
            # Further split large sections into smaller chunks
            section_chunks = self._split_section(section_content)

            for idx, chunk_content in enumerate(section_chunks):
                chunk = DocumentChunk(
                    content=chunk_content.strip(),
                    module=module_name,
                    chapter=chapter_name,
                    section=section_title,
                    file_path=str(file_path),
                    chunk_index=idx
                )
                chunks.append(chunk)

        return chunks

    def _split_by_headers(self, content: str) -> List[tuple[str, str]]:
        """
        Split content by h2 headers (##).

        Returns:
            List of (section_title, section_content) tuples
        """
        # Split by ## headers
        sections = []
        current_section = ""
        current_title = "Introduction"

        lines = content.split('\n')
        for line in lines:
            if line.startswith('## '):
                # Save previous section
                if current_section:
                    sections.append((current_title, current_section))

                # Start new section
                current_title = line.replace('##', '').strip()
                current_section = ""
            else:
                current_section += line + '\n'

        # Add last section
        if current_section:
            sections.append((current_title, current_section))

        return sections

    def _split_section(self, content: str) -> List[str]:
        """
        Split a section into smaller chunks if it exceeds max_chunk_size.

        Args:
            content: Section content

        Returns:
            List of chunk strings
        """
        if len(content) <= self.max_chunk_size:
            return [content]

        chunks = []
        current_chunk = ""

        # Split by paragraphs
        paragraphs = content.split('\n\n')

        for para in paragraphs:
            # If adding this paragraph would exceed limit, save current chunk
            if len(current_chunk) + len(para) > self.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = para
                else:
                    # Single paragraph is too long, split by sentences
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) > self.max_chunk_size:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = sentence
                        else:
                            current_chunk += " " + sentence
            else:
                current_chunk += "\n\n" + para

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks


def main():
    """Test the document chunker."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python document_chunker.py <docs_path>")
        sys.exit(1)

    docs_path = sys.argv[1]
    chunker = DocumentChunker(docs_path)
    chunks = chunker.chunk_documents()

    print(f"Total chunks created: {len(chunks)}")
    print("\nSample chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Module: {chunk.module}")
        print(f"Chapter: {chunk.chapter}")
        print(f"Section: {chunk.section}")
        print(f"Content (first 200 chars): {chunk.content[:200]}...")


if __name__ == "__main__":
    main()
