| model | group | COMMA_F1 | PERIOD_F1 | QUESTION_F1 | micro_P | micro_R | micro_F1 | macro_F1 | SER |
|---|---|---|---|---|---|---|---|---|---|
| Rule: period at end only | baseline | 0.0 | 19.4 | 0.0 | 97.0 | 5.7 | 10.8 | 6.5 | 94.3 |
| XLM-R base punct. model (zero-shot) | pre-trained | 62.3 | 85.4 | 83.7 | 74.1 | 76.6 | 75.4 | 77.1 | 40.5 |
| XLM-R large FullStop (zero-shot) | pre-trained | 64.8 | 86.8 | 87.7 | 76.5 | 78.3 | 77.4 | 79.8 | 36.4 |
| XLM-R base punct. model + domain FT | fine-tuned | 66.8 | 87.3 | 85.8 | 80.6 | 77.4 | 78.9 | 80.0 | 32.7 |
| DistilBERT + domain FT | fine-tuned | 57.3 | 79.1 | 71.3 | 72.8 | 67.4 | 70.0 | 69.2 | 45.6 |
| DistilBERT FT + GPT-2 rescoring (generic) | fine-tuned + ext. LM | 60.1 | 82.3 | 73.2 | 76.1 | 70.1 | 73.0 | 71.9 | 40.9 |
| DistilBERT FT + GPT-2 rescoring (domain) | fine-tuned + ext. LM | 63.0 | 83.5 | 76.8 | 76.1 | 73.5 | 74.8 | 74.4 | 39.3 |
| XLM-R base FT + GPT-2 rescoring (generic) | fine-tuned + ext. LM | 68.7 | 87.5 | 85.1 | 79.4 | 79.7 | 79.5 | 80.5 | 32.3 |
| XLM-R base FT + GPT-2 rescoring (domain) | fine-tuned + ext. LM | 69.3 | 88.4 | 86.7 | 80.5 | 80.3 | 80.4 | 81.5 | 31.0 |
