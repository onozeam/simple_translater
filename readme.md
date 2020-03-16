# Simple Translater
A translater from English to Frensh using Pytorch.   
This library could be used as not only CLI but also a web application. A web application interface is implemeted in Rust.

## Model
[Universal Transformers](https://arxiv.org/abs/1807.03819)

## Dataset
[European Parliament Proceedings Parallel Corpus](https://www.statmt.org/europarl/)

## Usage
### Trainning model or downloading optimized model state.
You can use floydhub for trainnig. this training will finished with in 1 hour.
```
$ pwd
.../simple_translater/universal_transformer
$ floyd init [any project name]
...
$ floyd run
```
After you finished trainning, you can download `model_state` file from "https://www.floydhub.com/[your_name]/projects/[project_name]/1/files/saved/model_state", and must put on `.../simple_translater/universal_transformer/saved/.` on your local computer.

If you don't want to train a model now, you can download `model_state` file which was already trained, from [my floydhub repository](https://www.floydhub.com/maezono/projects/universal_transformer/1/files/saved/model_state). And please put that file on `.../simple_translater/universal_transformer/saved/.` on your computer.

### Prediction on CLI
If you haven't used french and english on spacy, you need downdloading these data.
```
$ python3 -m spacy download fr
$ python3 -m spacy download en
```
Translate any target sentence, for example "we can use a computer."
```
$ python3 translate.py -sentence "we can use a computer."
nous pouvons utiliser un ordinateur.
```

### Prediction on a web appilication
Start up a web application,
```
$ pwd
.../simple_translater
$ cargo run --features watch
```
and access to [http://localhost:3000/](http://localhost:3000/)


## Versions
- Python: 3.7.6
- rustc: 1.38.0-nightly (To use [pyo3](https://github.com/PyO3/pyo3), we need nightly rust.)
