"""
Rewritten legal document chunk extractor with 4-thread processing.

This script processes XML legal documents concurrently (using 4 threads) to extract every element
that represents the lowest level of the legal hierarchy. Typically, it finds <subsection> elements.
However, if a <section> or <article> element has no nested <subsection>, then that entire section/article
is treated as a chunk.
For every chunk, the script builds a full heading by recursively collecting headings (from <num> and <heading> tags)
found in its parent elements.
Paragraphs and subparagraphs inside a chunk are treated as its content.

NOTE: Sometimes <paragraph> and <subparagraph> are the lowest hierarchy elements without having a <section>
or <subsection> parent. In this updated version, such elements are also extracted as separate chunks.

XML hierarchy considered:

  preliminary (outside main hierarchy)

  Main hierarchy:
    part
    ↓
    division
    ↓
    subdivision
    ↓
    chapter (in some legislation)
    ↓
    section/article (primary level)
    ↓
    subsection (if present) – otherwise, the entire section/article is a chunk.
    ↓
    paragraph/subparagraph (treated as content unless they are the lowest level)
"""

import glob
import csv
import re
import os
import copy
import math  # Needed for splitting content into subchunks.
from bs4 import BeautifulSoup
import tiktoken
import concurrent.futures

# Allowed tags for building the heading chain.
ALLOWED_HEADING_TAGS = {
    "preliminary",
    "part",
    "division",
    "subdivision",
    "chapter",
    "section",
    "article",
    "subsection"
}

def clean_text(text):
    """
    Cleans the given text by removing tabs, newlines,
    and extra whitespace, leaving words separated by a single space.
    """
    return " ".join(text.split())

def get_title(soup):
    """
    Extracts the title from the XML document.
    
    It first looks for a <shortTitle> (or <docTitle>), then appends <docName> if available.
    If none is found, it either attempts to extract a title from the text or prompts the user.
    """
    title = ""
    if soup.find("shortTitle"):
        title = clean_text(soup.find("shortTitle").get_text())
    elif soup.find("docTitle"):
        title = clean_text(soup.find("docTitle").get_text())
    else:
        full_text = clean_text(str(soup))
        match = re.search(r"This Ordinance may be cited as the (.+?)\.", full_text)
        if match:
            title = match.group(1)
        else:
            title = input("Cannot extract name of Ordinance, manually enter name: ")
            title = clean_text(title)
    if soup.find("docName"):
        title += " " + clean_text(soup.find("docName").get_text())
    return clean_text(title)

def build_full_heading(element):
    """
    Builds a full heading for a given element by traversing its ancestors.
    
    For every ancestor whose tag is in ALLOWED_HEADING_TAGS, we attempt to find a <num>
    and/or <heading> element (using only direct children). If neither is found, the
    tag name (capitalized) is used. The headings are concatenated with " > " as the separator.
    """
    headings = []
    for ancestor in reversed(list(element.parents)):
        if ancestor.name and ancestor.name.lower() in ALLOWED_HEADING_TAGS:
            part = ""
            num_elem = ancestor.find("num", recursive=False)
            heading_elem = ancestor.find("heading", recursive=False)
            if num_elem:
                part += clean_text(" ".join(num_elem.stripped_strings))
            if heading_elem:
                if part:
                    part += " "
                part += clean_text(" ".join(heading_elem.stripped_strings))
            if not part:
                part = ancestor.name.capitalize()
            headings.append(part)
    # Now add the current element’s own heading information.
    current = ""
    num_elem = element.find("num", recursive=False)
    heading_elem = element.find("heading", recursive=False)
    if num_elem:
        current += clean_text(" ".join(num_elem.stripped_strings))
    if heading_elem:
        if current:
            current += " "
        current += clean_text(" ".join(heading_elem.stripped_strings))
    if not current:
        current = element.name.capitalize()
    headings.append(current)
    full_heading = " > ".join(headings)
    return clean_text(full_heading)

def extract_chunk_content(element):
    """
    Extracts text content from a chunk element while excluding any nested 
    <subsection> elements and any direct header parts (i.e. <num> and <heading>).
    
    A deep copy of the element is made so that nested chunks (if any) are removed,
    preventing duplication between a parent and its children.
    """
    element_copy = copy.deepcopy(element)
    for nested in element_copy.find_all("subsection"):
        nested.decompose()
    for tag in element_copy.find_all(["num", "heading"]):
        tag.decompose()
    content = element_copy.get_text(" ", strip=True)
    return clean_text(content)

def extract_chunks(soup):
    """
    Finds all elements that represent the lowest level of the legal hierarchy in the XML document.
    
    Modified behavior:
      - If a <subsection> is present, it is always considered the lowest level.
      - If a <section> or <article> contains no nested <subsection>, then the entire element is a chunk.
      - If a <section> or <article> contains nested <subsection>s, any direct content (outside any nested subsection)
        is extracted as an additional chunk.
      - Additionally, if <paragraph> or <subparagraph> elements occur without being nested inside a 
        <section>, <article>, or <subsection>, they are treated as individual chunks.
    
    Returns a list of dictionaries with keys "heading" and "content".
    """
    chunks = []
    # Process <section>, <article>, and <subsection> elements.
    for elem in soup.find_all(["section", "article", "subsection"]):
        if elem.name.lower() == "subsection":
            # Always process subsections as lowest-level chunks.
            full_heading = build_full_heading(elem)
            content = extract_chunk_content(elem)
            chunks.append({
                "heading": full_heading,
                "content": content
            })
        elif elem.name.lower() in ["section", "article"]:
            # For a section or article that contains nested subsections, extract any direct content.
            parent_content = extract_chunk_content(elem)
            if elem.find("subsection"):
                # Only add the parent's own content if it exists.
                if parent_content.strip():
                    full_heading = build_full_heading(elem)
                    chunks.append({
                        "heading": full_heading,
                        "content": parent_content
                    })
            else:
                # No subsections: treat the whole element as a chunk.
                full_heading = build_full_heading(elem)
                chunks.append({
                    "heading": full_heading,
                    "content": parent_content
                })
    
    # NEW ADDITION:
    # Extract <paragraph> and <subparagraph> if they are not nested within a section/article/subsection.
    for elem in soup.find_all(["paragraph", "subparagraph"]):
        if not elem.find_parent(["section", "article", "subsection"]):
            full_heading = build_full_heading(elem)
            content = extract_chunk_content(elem)
            chunks.append({
                "heading": full_heading,
                "content": content
            })
    return chunks

def add_metadata_to_chunks(file_name, doc_title, chunks, encoder, start_id=0):
    """
    Adds an ID to each chunk and combines the document title, the chunk's full heading,
    and its content into one text field.
    
    For chunks that have more than 800 tokens, the content is naively split into smaller chunks,
    but each new chunk still includes the document title and heading.
    
    The CSV rows will have two headers: id and text.
    
    The id is assigned using a simple increment, starting from 'start_id'.
    Returns a tuple (new_chunks, next_available_id).
    """
    new_chunks = []
    current_id = start_id
    for chunk in chunks:
        header_str = f"{doc_title} {chunk['heading']}".strip()
        combined_text = f"{header_str} {chunk['content']}".strip()
        token_count = len(encoder.encode(combined_text))
        
        if token_count > 800:
            # Naively split the content while preserving the header in each sub-chunk.
            header_tokens = encoder.encode(header_str)
            header_token_count = len(header_tokens)
            allowed_tokens_for_content = 800 - header_token_count
            # Fallback if the header itself is too long.
            if allowed_tokens_for_content < 1:
                allowed_tokens_for_content = 800
            
            content_tokens = encoder.encode(chunk['content'])
            num_subchunks = math.ceil(len(content_tokens) / allowed_tokens_for_content)
            
            for j in range(num_subchunks):
                sub_tokens = content_tokens[j * allowed_tokens_for_content : (j+1) * allowed_tokens_for_content]
                # Decode this portion of content back to text.
                sub_content = encoder.decode(sub_tokens)
                sub_combined_text = f"{header_str} {sub_content}".strip()
                new_chunks.append({
                    "id": str(current_id),
                    "text": sub_combined_text
                })
                current_id += 1
        else:
            new_chunks.append({
                "id": str(current_id),
                "text": combined_text
            })
            current_id += 1
    return new_chunks, current_id

def write_chunks_to_csv(chunks, output_file, write_header=False):
    """
    Writes the list of chunks to a CSV file with headers: id and text.
    
    If write_header is True, it writes the header line; otherwise, it appends to the file.
    """
    mode = 'w' if write_header else 'a'
    with open(output_file, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text"])
        if write_header:
            writer.writeheader()
        writer.writerows(chunks)

def process_file(file_name, encoder):
    """
    Processes one XML file: reads and parses the file, extracts the document title,
    extracts the lowest-level chunks, and then uses add_metadata_to_chunks to add
    document-level metadata and split chunks if needed.

    Returns the list of processed chunks (without any file-specific IDs).
    """
    with open(file_name, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "xml")
    print("Processing", file_name)
    doc_title = get_title(soup)
    
    # Extract the lowest-level chunks (and any parent's own content)
    chunks = extract_chunks(soup)
    # Process the chunks to combine metadata and handle token splitting.
    # Starting ID is set to 0 for each file here, but we will reassign 
    # unique IDs later in the main thread.
    processed_chunks, _ = add_metadata_to_chunks(file_name, doc_title, chunks, encoder, start_id=0)
    # Remove the temporary 'id' field since we'll assign global IDs later.
    for chunk in processed_chunks:
        if "id" in chunk:
            del chunk["id"]
    return processed_chunks

def main():
    output_file = "corpus_leg.csv"
    # Create the encoder for model "gpt-4o" using tiktoken.
    encoder = tiktoken.encoding_for_model("gpt-4o")
    
    # Find all XML files in the specified directory.
    files = glob.glob("output/*.xml")
    
    # Use a thread pool with 4 threads to process files concurrently.
    all_chunks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Map each file to the process_file function.
        future_to_file = {executor.submit(process_file, file_name, encoder): file_name for file_name in files}
        for future in concurrent.futures.as_completed(future_to_file):
            file_name = future_to_file[future]
            try:
                chunks = future.result()
                all_chunks.extend(chunks)
                print(f"Finished processing {file_name}. Extracted {len(chunks)} chunks.")
            except Exception as exc:
                print(f"File {file_name} generated an exception: {exc}")
    
    # Reassign sequential global IDs for all collected chunks.
    for idx, chunk in enumerate(all_chunks):
        chunk["id"] = str(idx)
    
    # Write all extracted chunks to a CSV file (one header, then all rows).
    write_chunks_to_csv(all_chunks, output_file, write_header=True)
    print(f"Finished writing {len(all_chunks)} chunks to output.csv.")

if __name__ == "__main__":
    main()
