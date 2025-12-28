mkdir files_hotpotqa_t/
mkdir files_hotpotqa_v/

pip install -U pip
pip install -U -r requirements.txt

python analysis_shortanswer.py hotpotqa
python train_shortanswer.py hotpotqa
python analysis_shortanswer_patched.py hotpotqa

python report_metrics.py hotpotqa
python report_metrics_patched.py hotpotqa