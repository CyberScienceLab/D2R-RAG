setup.sh

python create_knowledge_base.py fever

python train_patcher.py fever
tar czf out.tar.gz out/

# tar xzvf out.tar.gz
python main.py fever
python report_metrics_prepatch.py fever
python report_metrics_postpatch.py fever
python report_deltas.py fever