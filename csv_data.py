import os
import re
import csv
import argparse
from pathlib import Path
from typing import List, Tuple

# LangChain imports
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


def extract_metadata(file_path: Path, content: str) -> Tuple[str, str]:
    """Extract ID from filename and Title from content."""
    # Get ID from filename (without extension)
    doc_id = file_path.stem

    # Extract title from content
    title_match = re.search(r'Title:\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
    title_value = title_match.group(1).strip() if title_match else "Unknown"

    print(f"Extracted ID: {doc_id}, Title: {title_value}")

    return doc_id, title_value


def create_digital_banking_chain():
    """Create a LangChain to analyze if content is related to digital banking."""
    # Initialize Ollama LLM
    llm = Ollama(model="deepseek-r1:latest")

    # Create prompt template
    prompt_template = PromptTemplate(
        input_variables=["document_content"],
        template="""
        Your task is to determine if the following document is related to digital banking development (backend API implementation, frontend implementation, or server infrastructure).

        A digital banking platform refers to technology systems that enable banking services through digital channels, including:
        - Online banking websites and mobile applications for customers
        - API gateways and services for financial transactions
        - Authentication and security systems
        - Account management systems
        - Payment processing infrastructure
        - Database systems storing financial records
        - Cloud or on-premises infrastructure supporting these services
        - UI/UX components specific to banking applications

        For your response, you MUST FOLLOW THESE EXACT INSTRUCTIONS:
        1. Start your response with ONLY "YES" or "NO" (in capital letters)
        2. After that single word, provide a brief 2-3 sentence summary of what the document is about
        3. Do not include any other information, analysis, or questions
        4. Do not use markdown formatting
        5. Keep your entire response under 100 words
        6. If you are unsure or if the document is so vague, respond with "SKIP" and do not provide any summary

        ```
        {document_content}
        ```
        """
    )

    # Create chain
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain


def clean_response(text: str) -> str:
    """Clean the response by removing any <think> tags and their content."""
    # Remove <think>...</think> blocks
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # Remove any other HTML/XML tags
    cleaned = re.sub(r'<[^>]+>', '', cleaned)

    return cleaned.strip()


def process_markdown_content(chain, content: str) -> Tuple[bool, str]:
    """Process markdown content to check if it's related to digital banking."""
    # Use generate instead of run
    response = chain.generate([{"document_content": content}])
    
    # Access the processed text from the generation results
    # The response structure is more complex with generate()
    generation = response.generations[0][0]
    response_text = generation.text
    
    # Extract the yes/no answer and summary
    response_upper = response_text.strip().upper()
    if response_upper.startswith("YES"):
        is_related = True
    elif response_upper.startswith("NO"):
        is_related = False
    else:
        # Handle cases where response doesn't start with YES/NO
        is_related = "YES" in response_upper[:20]  # Check first few characters

    return is_related, clean_response(response_text)


def load_markdown_files(directory: str) -> List[Path]:
    """Load all markdown files from the specified directory."""
    markdown_files = list(Path(directory).glob("*.md"))

    # If no markdown files are found, also try txt files
    if not markdown_files:
        markdown_files = list(Path(directory).glob("*.txt"))
        print(
            f"No .md files found, using {len(markdown_files)} .txt files instead")

    return markdown_files


def process_markdown_files(directory: str, output_file: str):
    """Process all markdown files in the directory and create a CSV file."""
    markdown_files = load_markdown_files(directory)

    if not markdown_files:
        print(f"No markdown or text files found in {directory}")
        return

    print(f"Found {len(markdown_files)} files to process")

    # Create digital banking analysis chain
    chain = create_digital_banking_chain()

    # Create CSV file and write header
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['ID', 'Title', 'Summary'])

        # Process each file
        for file_path in markdown_files:
            print(f"\nProcessing {file_path.name}...")

            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                # Extract metadata using filename as ID
                doc_id, title = extract_metadata(file_path, content)

                # Check if related to digital banking
                is_related, summary = process_markdown_content(chain, content)

                # Add to CSV
                csv_writer.writerow([doc_id, title, summary])
                print(f"Added to CSV: ID={doc_id}, Title={title}")
                print(f"Related to digital banking: {is_related}")
                print(f"Summary: {summary}")

            except Exception as e:
                print(f"Error processing {file_path.name}: {str(e)}")
                import traceback
                traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='Process markdown files to identify digital banking related content')
    parser.add_argument('--directory', '-d', required=True,
                        help='Directory containing markdown files')
    parser.add_argument(
        '--output', '-o', default='digital_banking_docs.csv', help='Output CSV file')
    args = parser.parse_args()

    process_markdown_files(args.directory, args.output)
    print(f"Processing complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()
