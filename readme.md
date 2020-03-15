# Simple Translater
A translater from English to Frensh using Pytorch.   
We can use this library as not only CLI but also a web application which is implemeted in Rust.

## Model
[Universal Transformers](https://arxiv.org/abs/1807.03819)

## Dataset
[European Parliament Proceedings Parallel Corpus](https://www.statmt.org/europarl/)

## Usage
### Trainning model or downloading optimized model state.
you can use floydhub for trainnig. this training will finished with in 1 hour.
```
$ pwd
.../simple_translater/universal_transformer/
$ floyd run
```

if you don't want to train a model now, you can download `model_state` which was already trained, from my floydhub repository.
model_state should be put at `.../simple_translater/universal_transformer/saved/model_state`

### Prediction on CLI
if you haven't use french and english on spacy, you need downdloading these data.
```
$ python3 -m spacy download fr
$ python3 -m spacy download en
```
translate any target sentence, for example "we can use a computer."
```
$ python3 translate.py -sentence "we can use a computer."
```

### Running a web appilication
start up a web application,
```
$ pwd
.../simple_translater
$ cargo run
```
and access to [http://localhost:3000/](http://localhost:3000/)


# Versions
Python: 
Rust: 
