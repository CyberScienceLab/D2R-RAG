mkdir files_fever_t/
mkdir files_fever_v/
mkdir files_hotpotqa_t/
mkdir files_hotpotqa_v/
mkdir files_fever_ts_t/
mkdir files_fever_ts_v/

pip install -U pip
pip install -U -r requirements.txt

# wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz
# gzip -d biencoder-nq-train.json.gz
# wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz
# gzip -d biencoder-nq-dev.json.gz
# wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
# gzip -d psgs_w100.tsv.gz

wget https://fever.ai/download/fever/shared_task_dev.jsonl

wget https://fever.ai/download/fever/wiki-pages.zip
unzip /content/wiki-pages.zip