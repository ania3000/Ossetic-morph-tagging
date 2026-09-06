# Ossetic Morphological Analysis

Official repository for Ossetic morphological tagging (in classic and multi-task modes) and paradigm-based lemmatization using BERT fine-tuning.

[![ArXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/2607.04895) <!-- Подставьте ссылку на arXiv -->
[![Tagger (Classic)](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Tagger-blue)](https://huggingface.co/ossetic-encoders/ossbert-morph-v2) <!-- Подставьте ссылку на профиль HF -->
[![Lemmatizer](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Lemmatizer-blue)](https://huggingface.co/ossetic-encoders/ossbert-lemm-v2) <!-- Подставьте ссылку на профиль HF -->
[![Dataset](https://img.shields.io/badge/GitHub-Ossetic--COT%20Dataset-green)](https://github.com/ania3000/Ossetic-COT)

## Overview

This repository provides models and code for Morphological Analysis (Tagging and Lemmatization) for Iron Ossetic. 

| Resource | Description | Link |
| :--- | :--- | :--- |
| **Pre-trained LM** | Google's multilingual BERT fine-tuned on Ossetic data. | [HuggingFace](https://huggingface.co/AlexeySorokin/ossbert-onc-unlab-from_multilingual-bs64-5epochs) |
| **Tagger** | Fine-tuned model for POS and features prediction in classic mode. | [HuggingFace](https://huggingface.co/ossetic-encoders/ossbert-morph-v2) |
| **Lemmatizer** | Fine-tuned model for abstract paradigm label prediction. | [HuggingFace](https://huggingface.co/ossetic-encoders/ossbert-lemm-v2) |
| **Dataset** | Morphologically annotated dataset adhering to UD schema (5454 sentences). | [GitHub](https://github.com/ania3000/Ossetic-COT) |
| **Paper** | Research paper describing methodology and experimental results. | [arXiv](https://arxiv.org/abs/2607.04895) |

### Demo HF Spaces

| Task | Description | Link |
| :--- | :--- | :--- |
|**MLM demo**| Filling masks in sentences in Ossetic. |[HF Spaces](https://huggingface.co/spaces/ania3000/ossetic-mlm) |
|**Tagger and lemmatizer demo**| Tagging and lemmatization of raw sentences in Ossetic. | [HF Spaces](https://huggingface.co/spaces/ania3000/ossetic-morph-analyzer) |


## Installation
```bash
git clone https://github.com/ania3000/Ossetic-morph-tagging.git
cd Ossetic-morph-tagging
bash setup_environment.sh
```

For tagging in multi-task mode or lemmatization change the 'mode' argument value in the code below to multitask or lemmatization.
   
```bash
python3 train.py \
    --mode classic \
    --model_checkpoint AlexeySorokin/ossbert-onc-unlab-from_multilingual-bs64-5epochs \
    --train_path data/train.conllu \
    --dev_path data/dev.conllu \
    --test_path data/test.conllu \
    --epochs 25
```
## Repository Structure

```text
Ossetic-morph-tagging/
├── .gitignore             
├── README.md               
├── requirements.txt        
├── setup_environment.sh    # Скрипт сборки pyparadigm и клонирования данных
├── train.py                # Единый скрипт для обучения (classic, multitask, lemmatization)
└── src/                    # Модули исходного кода
    ├── utils.py            # Чтение CoNLL-U, расчёт LCS и восстановление лемм по метке парадигмы
    ├── dataset.py          # Классы Dataset и DataCollator
    ├── models.py           # Архитектура Multi-Task модели
    └── metrics.py          # Расчет метрик
```

## Metrics
Evaluation metrics obtained on Ossetic-COT (version 2) test set:

|| Accuracy | Sentence Accuracy |
| :--- | :--- | :--- |
| **Tagger (classic mode)** | 95.90 | 62.02 |
| **Lemmatizer** | 98.17 | 80.73 |

## Citation
```
@inproceedings{shatskikh2026ossetic,
  title={Ossetic-COT: Designing a morphologically annotated corpus and morphological analyzer for Ossetic},
  author={Shatskikh, Anna and Sorokin, Alexey},
  booktitle={Computational Linguistics and Intellectual Technologies. Proceedings of the International Conference “Dialogue 2026”},
  volume={2026},
  year={2026}
}
```
