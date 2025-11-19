#!/usr/bin/env python3
"""
Test script for PDF download functionality
"""
import requests
from urllib.parse import quote
import os
import json

def test_pdf_download():
    """Test the PDF download functionality"""
    query = 'machine learning'
    pdf_dir = 'lifemodo_data/downloaded_pdfs'
    os.makedirs(pdf_dir, exist_ok=True)

    # Test arXiv API
    search_url = f'http://export.arxiv.org/api/query?search_query=all:{quote(query)}&start=0&max_results=1&sortBy=relevance&sortOrder=descending'

    try:
        print("🔍 Searching arXiv for PDFs...")
        response = requests.get(search_url, timeout=10)
        response.raise_for_status()

        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.content)

        for entry in root.findall('.//{http://www.w3.org/2005/Atom}entry')[:1]:
            title_elem = entry.find('.//{http://www.w3.org/2005/Atom}title')
            id_elem = entry.find('.//{http://www.w3.org/2005/Atom}id')

            if title_elem is not None and id_elem is not None:
                title = title_elem.text.strip()
                arxiv_id = id_elem.text.split('/')[-1]
                pdf_url = f'https://arxiv.org/pdf/{arxiv_id}.pdf'

                print(f'📄 Found: {title[:50]}...')
                print(f'🔗 URL: {pdf_url}')

                # Download
                print("📥 Downloading PDF...")
                pdf_response = requests.get(pdf_url, timeout=30)
                if pdf_response.status_code == 200:
                    pdf_filename = f'arxiv_{arxiv_id}.pdf'
                    pdf_path = os.path.join(pdf_dir, pdf_filename)

                    with open(pdf_path, 'wb') as f:
                        f.write(pdf_response.content)

                    print(f'✅ Downloaded: {pdf_path}')
                    print(f'📊 File size: {len(pdf_response.content)} bytes')
                    return True
                else:
                    print(f'❌ HTTP Error: {pdf_response.status_code}')
                    return False

        print('⚠️ No results found')
        return False

    except Exception as e:
        print(f'❌ Error: {e}')
        return False

if __name__ == "__main__":
    print('🔍 Testing PDF download functionality...')
    result = test_pdf_download()
    print(f'Result: {"✅ Success" if result else "❌ Failed"}')