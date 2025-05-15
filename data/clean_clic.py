import csv
from bs4 import BeautifulSoup
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

def splitData(original):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o",
        chunk_size=1024,
        chunk_overlap=0
    )
    texts = text_splitter.split_text(original)
    return texts

def convert_to_sentence_case(s):
    def process_segment(segment):
        if segment.isupper():
            return segment
        parts = segment.split('_')
        processed_parts = []
        for part in parts:
            words = re.split(r'(?<=[a-z])(?=[A-Z])', part)
            capitalized = [word.capitalize() for word in words]
            processed_parts.append(' '.join(capitalized))
        return ' '.join(processed_parts)
    
    segments = s.split('/')
    processed_segments = [process_segment(seg) for seg in segments]
    return '/'.join(processed_segments)

def main():
    input_file = 'clic.csv'
    output_file = 'corpus_clic.csv'
    counter = 0

    with open(input_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile_out:
            fieldnames = ['id', 'text']
            writer = csv.DictWriter(csvfile_out, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in reader:
                try:
                    link = row['Link']
                    print(f"Processing {link}")
                    html_content = row['Html']
                    soup = BeautifulSoup(html_content, 'html.parser')
                    divs = soup.find_all('div', class_='text-detail')
                    if len(divs) == 0:
                        continue
                    assert len(divs) == 1
                    title = (convert_to_sentence_case(link.replace("https://clic.org.hk/en/topics/","")).replace("/", " > ") + f"\n{divs[0].find('h2').get_text()}").strip()
                    text = divs[0].get_text().replace(divs[0].find('h2').get_text(), "")
                    text = "\n".join([line.strip() for line in text.split("\n") if line.strip() != ""])
                    texts = splitData(text)
                    for text in texts:
                        writer.writerow({'id': counter, 'text': f"{title}\n{text}"})
                        counter += 1
                except Exception as e:
                    print(f"Error: {e}")
                    continue

if __name__ == "__main__":
    main()