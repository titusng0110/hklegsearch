from bs4 import BeautifulSoup
import glob
import os

def check_and_clean_docstatus():
    # Get all XML files in output/ directory
    xml_files = glob.glob('output/*.xml')
    
    # If no XML files found, inform the user
    if not xml_files:
        print("No XML files found in output/ directory")
        return
    
    # Keep track of deleted files
    deleted_files = []
    
    # Process each XML file
    for xml_file in xml_files:
        try:
            # Read the XML file
            with open(xml_file, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Parse XML with BeautifulSoup
            soup = BeautifulSoup(content, 'xml')
            
            # Search for docstatus tag
            docstatus = soup.find('docStatus')
            
            # Check if docstatus exists and its value
            if docstatus and docstatus.text.strip() != "In effect":
                # Delete the file if docstatus is not "In effect"
                os.remove(xml_file)
                deleted_files.append(os.path.basename(xml_file))
                print(f"Deleted {os.path.basename(xml_file)} - docStatus: {docstatus.text.strip()}")
            else:
                print(f"Kept {os.path.basename(xml_file)} - docStatus: {docstatus.text.strip() if docstatus else 'Not found'}")
                
        except Exception as e:
            print(f"Error processing {xml_file}: {str(e)}")
    
    # Summary of operation
    if deleted_files:
        print(f"\nDeleted {len(deleted_files)} files:")
        for file in deleted_files:
            print(f"- {file}")
    else:
        print("\nNo files were deleted.")

if __name__ == "__main__":
    check_and_clean_docstatus()