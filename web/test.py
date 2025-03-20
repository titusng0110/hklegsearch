import requests
import json

def fetch(url, query):
    try:
        params = {'payload': query}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching content from {url}: {str(e)}")
        return None

def main():
    url = "http://127.0.0.1:30000/api/"
    while True:
        input_text = input("Ask: ").strip()
        if input_text == 'exit':
            break
        result = fetch(url, query=input_text)
        if isinstance(result, dict) or isinstance(result, list):
            print(json.dumps(result, indent=2))
        else:
            print(result)

if __name__ == '__main__':
    main()