# LSI Model

Uses Latent Semantic Indexing to find most and least similar talks

## Usage

All talks should be stored in `models/lsi/data/` in `.txt` format.

In the console navigate to the `lsi` directory and run the python script with all talks to analyze for similarity as arguments:

```bash
cd models/lsi
python lsi.py ./data/1001.txt ./data/1006.txt ...
```

Keep `stopwords.txt` checked in to git so we are consistent.
