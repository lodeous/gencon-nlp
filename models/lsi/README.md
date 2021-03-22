# LSI Model

Uses Latent Semantic Indexing to find most and least similar talks

## Usage

All talks should be stored in `.txt` format.

In the console navigate to the `lsi` directory and run the python script with all talks to analyze for similarity as arguments:

```python
paths = ['data/all.txt', 'the/document.txt', 'paths/will.txt', 'go/here.txt']
lsi = LSI(n_components=8)
lsi.fit(paths)
cross_similarities = lsi.cross_similarity(['data/all.txt', 'paths/will.txt'])
```

Keep `stopwords.txt` checked in to git so we are consistent.
