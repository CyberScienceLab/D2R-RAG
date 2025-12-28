mkdir files_hotpotqa_ts_t/
mkdir files_hotpotqa_ts_v/

pip install -U pip
pip install -U -r requirements.txt

python analysis_shortanswer.py hotpotqa_ts
python train_shortanswer.py hotpotqa_ts
python analysis_shortanswer_patched.py hotpotqa_ts

python report_metrics.py hotpotqa_ts
python report_metrics_patched.py hotpotqa_ts