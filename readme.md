## Setup
download the t4sa raw_tweets_text.csv and the t4sa_text_sentiment.txv into data

setup a venv and run pip install requirements.txt

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run `./custom_MLP/prep.ipynb` to create `../project_data/t4sa_data.csv`

## Demo 

Run `./custom_MLP/demo.py` for the demo of the implemented model on the subset of t4sa dataset in `./custom_MLP/custom_mlp.py`