mkdir files_hotpotqa_selfrag_t/
mkdir files_hotpotqa_selfrag_v/

pip install -U pip
pip install -U -r requirements.txt

python download_selfrag_pack.py

pip install -q huggingface-hub
huggingface-cli download m4r1/selfrag_llama2_7b-GGUF selfrag_llama2_7b.q4_k_m.gguf --local-dir . --local-dir-use-symlinks False

# To run Llama-cpp on cuda: (takes about 30 mins on colab)
CMAKE_ARGS="-DLLAMA_CUDA=on" \
FORCE_CMAKE=1 \
pip install llama-cpp-python --upgrade

python analysis_shortanswer_patched_selfrag.py hotpotqa

python report_metrics.py hotpotqa_selfrag
python report_metrics_patched.py hotpotqa_selfrag