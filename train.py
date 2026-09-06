import argparse
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    DataCollatorForTokenClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)

from src.utils import read_conllu, read_mt_conllu, get_all_tasks
from src.dataset import UDDataset, MultiTaskUDDataset, MultiTaskDataCollator
from src.models import AutoModelForMultiTaskTokenClassification
from src.metrics import make_compute_metrics, make_compute_multitask_metrics


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, use_fast=True, add_prefix_space=True)

    if args.mode == "lemmatization":
        print("=== Запуск лемматизации ===")
        from src.utils import read_lemmatization_conllu
        from src.dataset import LemmatizationDataset
        from src.metrics import make_compute_lemmatization_metrics

        train_data = read_lemmatization_conllu(args.train_path)
        dev_data = read_lemmatization_conllu(args.dev_path)
        test_data = read_lemmatization_conllu(args.test_path)

        train_ds = LemmatizationDataset(train_data, tokenizer)
        dev_ds = LemmatizationDataset(dev_data, tokenizer, tags=train_ds.tags_)
        test_ds = LemmatizationDataset(test_data, tokenizer, tags=train_ds.tags_)

        model = AutoModelForTokenClassification.from_pretrained(
            args.model_checkpoint, 
            num_labels=len(train_ds.tags_)
        )
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        compute_fn = make_compute_lemmatization_metrics(dev_ds)
        metric_name = "Lemma_Accuracy"

    if args.mode == "multitask":
        print("=== Запуск классификации в режиме Multi-Task ===")
        task_names = get_all_tasks(args.train_path)
        
        train_data = read_mt_conllu(args.train_path, task_names)
        dev_data = read_mt_conllu(args.dev_path, task_names)
        test_data = read_mt_conllu(args.test_path, task_names)

        train_ds = MultiTaskUDDataset(train_data, tokenizer)
        dev_ds = MultiTaskUDDataset(dev_data, tokenizer, tags=train_ds.tags_)
        test_ds = MultiTaskUDDataset(test_data, tokenizer, tags=train_ds.tags_)

        num_labels_dict = {task: len(train_ds.tags_[task]) for task in train_ds.tasks}
        model = AutoModelForMultiTaskTokenClassification(args.model_checkpoint, num_labels_dict)

        data_collator = MultiTaskDataCollator(tokenizer)
        compute_fn = make_compute_multitask_metrics(dev_ds, train_ds.tags_)
        metric_name = "Full_Tag_Accuracy"

    else:
        print("=== Запуск классификации в режиме Single-Task (Classic) ===")
        train_data = read_conllu(args.train_path)
        dev_data = read_conllu(args.dev_path)
        test_data = read_conllu(args.test_path)

        train_ds = UDDataset(train_data, tokenizer)
        dev_ds = UDDataset(dev_data, tokenizer, tags=train_ds.tags_)
        test_ds = UDDataset(test_data, tokenizer, tags=train_ds.tags_)

        model = AutoModelForTokenClassification.from_pretrained(
            args.model_checkpoint, 
            num_labels=len(train_ds.tags_)
        )
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        compute_fn = make_compute_metrics(dev_ds)
        metric_name = "Accuracy"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        optimizers=(AdamW(model.parameters(), lr=args.lr, weight_decay=0.01), None),
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.train()

    print("Тестирование...")
    if args.mode == "multitask":
        trainer.compute_metrics = make_compute_multitask_metrics(test_ds, train_ds.tags_)
    else:
        trainer.compute_metrics = make_compute_metrics(test_ds)

    results = trainer.predict(test_ds)
    print("Test Results:", results.metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["classic", "multitask", "lemmatization"], default="classic")
    parser.add_argument("--model_checkpoint", type=str, default="AlexeySorokin/ossbert-onc-unlab-from_multilingual-bs64-5epochs")
    parser.add_argument("--train_path", type=str, default="data/train.conllu")
    parser.add_argument("--dev_path", type=str, default="data/dev.conllu")
    parser.add_argument("--test_path", type=str, default="data/test.conllu")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=5e-5)
    
    args = parser.parse_args()
    main(args)
