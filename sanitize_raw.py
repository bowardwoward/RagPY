import os
import re
import glob
from pathlib import Path
import shutil

def clean_md_for_rag(md_text):
    """
    Clean markdown text from BSP circulars and memorandums for RAG embeddings.
    
    Args:
        md_text (str): Raw markdown text
        
    Returns:
        tuple: (cleaned_text, doc_id) - Cleaned text with metadata and document ID
    """
    # Extract document ID from various patterns
    doc_id_match = re.search(r'(?:CIRCULAR LETTER NO\.|MEMORANDUM NO\.|#)\s*([\w\-\.]+)', md_text)
    if not doc_id_match:
        # Try alternative format (e.g., M-2020-071)
        doc_id_match = re.search(r'#\s*([CM]\-\d{4}\-\d{3})', md_text)
    
    doc_id = doc_id_match.group(1).strip() if doc_id_match else "UNKNOWN"
    
    # Remove any spaces in the document ID
    doc_id = re.sub(r'\s+', '', doc_id)
    
    # Clean up the text
    cleaned_text = md_text
    
    # Remove markdown headings while preserving content
    cleaned_text = re.sub(r'#{1,6}\s+', '', cleaned_text)
    
    # Remove page markers
    cleaned_text = re.sub(r'## Page \d+', '\n', cleaned_text)
    
    # Remove unnecessary whitespace, including multiple spaces and newlines
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    cleaned_text = re.sub(r' {2,}', ' ', cleaned_text)
    
    # Remove digital signature artifacts
    cleaned_text = re.sub(r'Digitally signed by.*?\+\d{2}\'\d{2}\'', '', cleaned_text)
    
    # Remove strange artifacts like "eee @ ee e" or ". . a S'S n ee"
    cleaned_text = re.sub(r'[e\.]+\s*[@\.]\s*[e\.]+\s*[e\.]+', '', cleaned_text)
    cleaned_text = re.sub(r'\.\s*\.\s*[a-zA-Z\'\s]+', '', cleaned_text)
    
    # Extract title/subject
    subject_match = re.search(r'Subject\s*:\s*(.+?)(?=\n)', cleaned_text)
    title = subject_match.group(1).strip() if subject_match else "No Subject"
    
    # Extract date
    date_match = re.search(r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})', cleaned_text)
    date = date_match.group(1) if date_match else ""
    
    # Extract issuer (person who signed)
    issuer_match = re.search(r'([A-Z\.\s]+)\nDeputy Governor|Managing Director', cleaned_text)
    issuer = issuer_match.group(1).strip() if issuer_match else ""
    
    # Create enhanced metadata section
    metadata = f"# {doc_id}\n"
    # metadata += f"# {title}\n"
    metadata += "\n"
    
    # Strip excess whitespace from the content
    cleaned_text = cleaned_text.strip()
    
    # Combine metadata with cleaned content
    final_text = metadata + cleaned_text
    
    return final_text, doc_id

def process_md_folder(input_folder):
    """
    Process all markdown files in a folder and its subfolders.
    Clean content and rename files based on document ID.
    
    Args:
        input_folder (str): Path to the folder containing markdown files
    """
    # Get all markdown files
    md_files = glob.glob(os.path.join(input_folder, "**/*.md"), recursive=True)
    
    print(f"Found {len(md_files)} markdown files to process")
    
    successful = 0
    failed = 0
    skipped = 0
    
    for file_path in md_files:
        try:
            # Read the original file
            with open(file_path, 'r', encoding='utf-8') as file:
                md_text = file.read()
            
            # Skip files that don't appear to be BSP documents
            if not re.search(r'BANGKO SENTRAL NG PILIPINAS|CIRCULAR LETTER|MEMORANDUM NO\.', md_text):
                print(f"Skipping non-BSP document: {file_path}")
                skipped += 1
                continue
            
            # Clean the content and get document ID
            cleaned_text, doc_id = clean_md_for_rag(md_text)
            
            if doc_id == "UNKNOWN":
                print(f"Warning: Could not extract document ID from {file_path}. Using original filename.")
                new_file_path = file_path
            else:
                # Create new filename with the document ID
                dir_path = os.path.dirname(file_path)
                new_file_name = f"{doc_id}.md"
                new_file_path = os.path.join(dir_path, new_file_name)
                
                # Check if the new filename is different from the original
                if os.path.normpath(new_file_path) != os.path.normpath(file_path):
                    # Check if file with new name already exists
                    if os.path.exists(new_file_path):
                        print(f"Warning: File {new_file_path} already exists. Using {doc_id}_duplicate.md")
                        new_file_name = f"{doc_id}_duplicate.md"
                        new_file_path = os.path.join(dir_path, new_file_name)
            
            # Write the cleaned content
            with open(new_file_path, 'w', encoding='utf-8') as file:
                file.write(cleaned_text)
            
            # If the filename changed, remove the original file
            if new_file_path != file_path and os.path.exists(file_path):
                os.remove(file_path)
                
            print(f"Processed: {os.path.basename(file_path)} → {os.path.basename(new_file_path)}")
            successful += 1
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            failed += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful} files")
    print(f"Skipped: {skipped} files")
    print(f"Failed: {failed} files")

def main():
    # Get the input folder from user
    input_folder = input("Enter the path to your markdown files folder: ").strip()
    
    # Validate folder exists
    if not os.path.isdir(input_folder):
        print(f"Error: The folder '{input_folder}' does not exist.")
        return
    
    # Confirm before proceeding
    confirm = input(f"This will clean up, modify and rename files in '{input_folder}' and its subfolders. Continue? (y/n): ").lower()
    if confirm != 'y':
        print("Operation cancelled.")
        return
    
    # Process the folder
    process_md_folder(input_folder)

if __name__ == "__main__":
    main()