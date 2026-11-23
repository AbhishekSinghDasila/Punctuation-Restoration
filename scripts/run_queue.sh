cd "$(dirname "$0")/.."
PY=venv/Scripts/python.exe
export PYTHONWARNINGS=ignore
have() { [ -f "results/preds/$1.pkl" ]; }
wait_model() { until [ -f "models/hf/$1/config.json" ] && { [ -f models/hf/$1/model.safetensors ] || [ -f models/hf/$1/pytorch_model.bin ]; }; do sleep 30; done; }

until have distilbert_finetuned; do sleep 30; done
have distilbert_finetuned+lm_domain || $PY -u scripts/04_external_lm.py --tagger distilbert_finetuned
wait_model kredor__punctuate-all
have punctall_zeroshot || $PY -u scripts/03_run_tagger.py --tag punctall_zeroshot --model kredor/punctuate-all
have punctall_finetuned || $PY -u scripts/03_run_tagger.py --tag punctall_finetuned --model kredor/punctuate-all --train --epochs 1 --lr 2e-5
have punctall_finetuned+lm_domain || $PY -u scripts/04_external_lm.py --tagger punctall_finetuned
wait_model oliverguhr__fullstop-punctuation-multilang-large
have fullstop_zeroshot || $PY -u scripts/03_run_tagger.py --tag fullstop_zeroshot --model oliverguhr/fullstop-punctuation-multilang-large
$PY -u scripts/05_evaluate_report.py
echo QUEUE_DONE
