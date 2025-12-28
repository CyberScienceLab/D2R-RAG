mkdir files_fever_selfrag_t/
mkdir files_fever_selfrag_v/

pip install -U pip
pip install -U -r requirements.txt

python download_selfrag_pack.py

pip install -q huggingface-hub
huggingface-cli download m4r1/selfrag_llama2_7b-GGUF selfrag_llama2_7b.q4_k_m.gguf --local-dir . --local-dir-use-symlinks False

# To run Llama-cpp on cuda: (takes about 30 mins on colab)
CMAKE_ARGS="-DLLAMA_CUDA=on" \
FORCE_CMAKE=1 \
pip install llama-cpp-python --upgrade

wget https://fever.ai/download/fever/shared_task_dev.jsonl

wget https://fever.ai/download/fever/wiki-pages.zip
unzip /content/wiki-pages.zip

python create_knowledge_base.py fever

python analysis_patched_selfrag.py fever

python report_metrics.py fever_selfrag
python report_metrics_patched.py fever_selfrag