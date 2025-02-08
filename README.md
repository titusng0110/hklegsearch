
# HK Legislation Smart Search

An AI-powered search engine for Hong Kong legislation, utilizing embeddings for semantic search capabilities.

## ⚠️ Disclaimer

This tool is designed for research purposes only and should not be relied upon as legal advice. Users should consult qualified legal professionals for specific legal matters. The accuracy and completeness of the search results cannot be guaranteed.

## 🌟 Features

- Semantic search across Hong Kong legislation with BGE-M3
- Preservation of legal document hierarchy and metadata
- Fast and efficient search using embeddings with Faiss
- Web-based interface for easy access

## 🚀 Quick Start

### Prerequisites

- a rtx 3000 series GPU with 8gb vram or better

### Installation (Linux)

1. Clone the repository
```bash
git clone https://github.com/titusng0110/hklegsearch.git
cd hklegsearch
```

2. Create and activate the Conda environment
```bash
conda env create -f environment.yml
conda activate hklegsearch  # assuming environment name is hklegsearch
```

## 📦 Data Processing Pipeline

### 1. Data Collection
```bash
cd data
./download.sh  # Downloads current Hong Kong legislation from data.gov.hk
```

### 2. Data Processing
```bash
./extract.sh   # Extracts and consolidates XML files
python clean.py      # Filters for legislation with status "In effect"
python big_clean.py  # Processes documents into searchable chunks
python embed.py      # Generates embeddings for semantic search
```

## 🌐 Web Interface

### 3. SSL Certificate Setup
Before starting the server, you'll need SSL certificates. You have two options:

#### Option A: Generate Self-Signed Certificates
```bash
# Generate self-signed certificates
cd ../web
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365
```

#### Option B: Use Your Own Certificates
Place your existing SSL certificates in the `web` directory:
- `cert.pem`: Your SSL certificate
- `key.pem`: Your private key

### 4. Starting the Server
```bash
# in web/ directory
python -m gunicorn -c gunicorn.conf.py
```

Once the server is running, access the search interface at:
```
https://localhost:30000
```

## 🛠️ Technical Stack

- **Language**: Python
- **Data Processing**: BeautifulSoup4, Polars, Parquet
- **Embeddings**: FlagEmbedding
- **Server**: Flask, Gunicorn
- **Data Source**: data.gov.hk

## 📝 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

- **Developer**: Titus Ng
- **Email**: titusngtszhin@gmail.com
- **GitHub**: [titusng0110](https://github.com/titusng0110)

For questions, suggestions, or collaboration opportunities, feel free to reach out!

---
Built by a law student with minor in CS
