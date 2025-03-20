import csv
import requests

def fetch_html(url):
    """Fetch HTML content from a URL with error handling."""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching content from {url}: {str(e)}")
        return f"Error fetching content: {str(e)}"

def process_csv(input_file, output_file):
    """Process CSV file and save results incrementally."""
    # Open output file and write header first
    with open(output_file, 'w', newline='', encoding='utf-8') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=['Id', 'Link', 'Html'])
        writer.writeheader()

        # Now process input file row by row
        with open(input_file, 'r', encoding='utf-8') as in_f:
            reader = csv.DictReader(in_f)
            for row in reader:
                html_content = fetch_html(row['Link'])
                # Write the result immediately
                writer.writerow({
                    'Id': row['Id'],
                    'Link': row['Link'],
                    'Html': html_content
                })

if __name__ == '__main__':
    process_csv('CLIC_Content_List.csv', 'clic.csv')