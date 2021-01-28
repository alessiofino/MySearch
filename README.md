# MySearch
This is a small hobby project I created to experiment with fast text search in documents.

This is a command line tool to perform fast text searches on JSON documents.
The code is partially based on [MeiliSearch](https://github.com/meilisearch/MeiliSearch).

<p align="center">
  <img src="assets/search.gif" alt="Web interface gif" />
</p>

## Features
* Search as-you-type (answers < 1 milliseconds)
* Full-text search
* Typo tolerant (understands typos and miss-spelling)

The results of searches are ordered first by increasing number of typos and then by proximity of the search terms in the document.

## Limitations
* The JSON document must contain an array of objects, the objects can have any structure, only the string fields are indexed.
* The tool doesn't make distinctions between the different fields in the documents, the fields are flattened in a single string during indexing.

## Getting started
Index a document
```
cargo run --release -- --index dataset/movies.json
```
If the index has already been created start searching using:
```
cargo run --release -- --attributes "title,id,genres,poster,overview"
```
Help
```
cargo run --release -- --help
```