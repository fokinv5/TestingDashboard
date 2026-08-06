import os

import fitz
import pymupdf
import re
import json

input_folder = "./ISTQB_sy"
output_folder = "./ISTQB_json"

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(input_folder, filename)
        json_filename = os.path.splitext(filename)[0] + ".json"
        json_path = os.path.join(output_folder, json_filename)

        doc = fitz.open(pdf_path)
        document_data = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text("text")
            lines = page_text.splitlines()
            #page_data = json.loads(page_text)
            document_data.append({
                "page": page_num + 1,
                "content": lines
            })
            with open(json_path, "w", encoding="utf-8") as json_file:
                json.dump(document_data, json_file, indent=4, ensure_ascii=False)

        print("Batch conversion completed!")
