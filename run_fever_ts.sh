mkdir files_fever_ts_t/
mkdir files_fever_ts_v/

pip install -U pip
pip install -U -r requirements.txt

wget https://fever.ai/download/fever/shared_task_dev.jsonl

wget https://fever.ai/download/fever/wiki-pages.zip
unzip /content/wiki-pages.zip

python create_knowledge_base.py fever

python analysis.py fever_ts
python train.py fever_ts
python analysis_patched.py fever_ts

python report_metrics.py fever_ts
python report_metrics_patched.py fever_ts