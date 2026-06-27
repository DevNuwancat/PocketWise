import pandas as pd 
import pdfplumber
import os 

pdf_path = 'commbank.pdf'

try:
    with pdfplumber.open(pdf_path) as pdf:
        print("It open successfully !")
        print("Number of Page : ", len(pdf.pages))

        first_page = pdf.pages[1]
        text = first_page.extract_text()

        print("\n-- TEXT ON PAGE 1 --")
        print(text)
except FileNotFoundError:
    print(f"Error: The file {pdf_path} was not found.")
except PermissionError:
    print(f"Error: Permission denied to open the file {pdf_path}.")
except Exception as e:
    print(f"An Expected Error Occurred: {e}")
