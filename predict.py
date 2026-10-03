import os
import argparse
import scipy.special
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.utils import rule_tokenize, restore_lemma, parse_label
from src.dataset import InferUDDataset

class BiLSTMCharTagger(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        return self.fc(lstm_out)

class WordSegmenterDecoder:
    def __init__(self, label_list):
        self.target_symbols_ = label_list
        self.target_symbol_codes_ = {sym: i for i, sym in enumerate(self.target_symbols_)}
        self.target_symbols_number_ = len(self.target_symbols_)

    def get_possible_next_states(self, prev_state_code):
        prev_label = self.target_symbols_[prev_state_code]
        START_LABELS = ["B", "S", "A", "N", "Z", "Y"]
        if prev_label in ["B", "A", "N", "Z", "Y", "I"]:
            next_labels = ["I", "E"]
        elif prev_label in ["E", "S", "X", "O"]:
            next_labels = START_LABELS + ["X", "O"]
        else:
            next_labels = []
        return [self.target_symbol_codes_[lbl] for lbl in next_labels if lbl in self.target_symbol_codes_]

    def is_correct_sequence(self, labels):
        if not labels or labels[0] in ["I", "E"] or labels[-1] not in ["E", "S", "X", "O"]:
            return False
        for i in range(len(labels) - 1):
            curr_code = self.target_symbol_codes_[labels[i]]
            next_code = self.target_symbol_codes_[labels[i+1]]
            if next_code not in self.get_possible_next_states(curr_code):
                return False
        return True

    def decode_best(self, probs, length):
        best_states = np.argmax(probs[:length], axis=1)
        best_labels = [self.target_symbols_[state_index] for state_index in best_states]
        if self.is_correct_sequence(best_labels):
            return best_states.tolist()
        costs, states = [], []
        first_step_costs = [np.inf] * self.target_symbols_number_
        first_step_states = [None] * self.target_symbols_number_
        for lbl in ["B", "S", "A", "N", "Z", "Y", "X", "O"]:
            if lbl in self.target_symbol_codes_:
                idx = self.target_symbol_codes_[lbl]
                first_step_costs[idx] = -np.log(probs[0, idx] + 1e-12)
                first_step_states[idx] = idx
        costs.append(first_step_costs)
        states.append(first_step_states)
        for i in range(1, length):
            state_order = np.argsort(costs[-1])
            curr_costs = [np.inf] * self.target_symbols_number_
            prev_states = [None] * self.target_symbols_number_
            for prev_state in state_order:
                if np.isinf(costs[-1][prev_state]): break
                for state in self.get_possible_next_states(prev_state):
                    if np.isinf(curr_costs[state]):
                        curr_costs[state] = costs[-1][prev_state] - np.log(probs[i, state] + 1e-12)
                        prev_states[state] = prev_state
            costs.append(curr_costs)
            states.append(prev_states)
        possible_final_states = [self.target_symbol_codes_[lbl] for lbl in ["E", "S", "X", "O"] if lbl in self.target_symbol_codes_]
        best_states_path = [min(possible_final_states, key=(lambda x: costs[-1][x]))]
        for j in range(length - 1, 0, -1):
            best_states_path.append(states[j][best_states_path[-1]])
        return best_states_path[::-1]

    def labels_to_words(self, input_string, labels):
        tokens, curr_token = [], ""
        for letter, label in zip(input_string, labels):
            if label == "O":
                if curr_token: tokens.append(curr_token); curr_token = ""
            elif label == "A":
                if curr_token: tokens.append(curr_token)
                curr_token = "æ" + letter
            elif label == "N":
                if curr_token: tokens.append(curr_token)
                curr_token = "и" + letter
            elif label == "Z":
                if curr_token: tokens.append(curr_token)
                curr_token = "ц"
            elif label == "Y":
                if curr_token: tokens.append(curr_token)
                curr_token = "и"
            elif label == "B":
                if curr_token: tokens.append(curr_token)
                curr_token = letter
            elif label == "S":
                if curr_token: tokens.append(curr_token)
                tokens.append(letter); curr_token = ""
            elif label in ["I", "E"]:
                curr_token += letter
                if label == "E": tokens.append(curr_token); curr_token = ""
            elif label == "X":
                if curr_token: tokens.append(curr_token); curr_token = ""
        if curr_token: tokens.append(curr_token)
        return tokens
        
class OsseticPipeline:
    def __init__(self, morph_model_name, lemm_model_name, segmenter_path=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(morph_model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(morph_model_name).to(self.device).eval()
        self.classes = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]

        self.tokenizer_l = AutoTokenizer.from_pretrained(lemm_model_name)
        self.model_l = AutoModelForTokenClassification.from_pretrained(lemm_model_name).to(self.device).eval()
        self.classes_l = [self.model_l.config.id2label[i] for i in range(len(self.model_l.config.id2label))]

        self.lstm_model = None
        if segmenter_path and os.path.exists(segmenter_path):
            checkpoint = torch.load(segmenter_path, map_location='cpu')
            self.char2idx = checkpoint['char2idx']
            self.LABEL_LIST = checkpoint['LABEL_LIST']
            config = checkpoint['model_config']
            
            self.lstm_model = BiLSTMCharTagger(
                vocab_size=len(self.char2idx),
                embedding_dim=config['embedding_dim'],
                hidden_dim=config['hidden_dim'],
                output_dim=len(self.LABEL_LIST)
            ).to(self.device).eval()
            self.lstm_model.load_state_dict(checkpoint['model_state_dict'])
            self.segmenter_decoder = WordSegmenterDecoder(self.LABEL_LIST)

    def run_lstm_segmentation(self, input_text):
        if not self.lstm_model or not input_text.strip():
            return input_text
        char_ids = [self.char2idx.get(c, self.char2idx.get("<UNK>", 0)) for c in input_text]
        input_tensor = torch.tensor([char_ids]).to(self.device)
        with torch.no_grad():
            logits = self.lstm_model(input_tensor)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        pred_codes = self.segmenter_decoder.decode_best(probs, len(input_text))
        pred_labels = [self.LABEL_LIST[code] for code in pred_codes]
        return " ".join(self.segmenter_decoder.labels_to_words(input_text, pred_labels))

    def predict_top_k(self, model, dataset, classes):
        model.eval()
        answer = []
        with torch.no_grad():
            for elem in dataset:
                inputs = {
                    "input_ids": torch.tensor(elem["input_ids"]).unsqueeze(0).to(self.device),
                    "attention_mask": torch.tensor(elem["attention_mask"]).unsqueeze(0).to(self.device)
                }
                logits = model(**inputs).logits.squeeze(0).cpu().numpy()
                mask = elem["mask"]
                probs = scipy.special.softmax(logits, axis=-1)[:len(mask)]
                top_k_indices = np.argsort(probs, axis=-1)[:, -1:][:, ::-1]
                
                top_k_labels = [classes[top_k_indices[i][0]] for i in range(len(mask)) if mask[i]]
                answer.append(top_k_labels)
        return answer

    def analyze_text(self, text):
        text = text.replace('Ӕ', 'Æ').replace('ӕ', 'æ')
        text = rule_tokenize([text])[0]
        if self.lstm_model:
            text = self.run_lstm_segmentation(text)

        words = text.split()
        if not words: return ""

        data_sample = {"words": words}
        test_dataset = InferUDDataset([data_sample], self.tokenizer, tags=self.classes)
        tag_preds = self.predict_top_k(self.model, test_dataset, self.classes)[0]

        test_dataset_l = InferUDDataset([data_sample], self.tokenizer_l, tags=self.classes_l)
        lemma_preds = self.predict_top_k(self.model_l, test_dataset_l, self.classes_l)[0]

        result = []
        for counter, (word, tag_label, lemma_label) in enumerate(zip(words, tag_preds, lemma_preds), start=1):
            lemma_str = restore_lemma(word, lemma_label).lower()
            upos, feats = parse_label(tag_label)
            if upos == "PROPN" and lemma_str:
                lemma_str = lemma_str[0].upper() + lemma_str[1:]

            conllu_line = [str(counter), word, lemma_str, upos, "_", feats, "_", "_", "_", "_"]
            result.append("\t".join(conllu_line))

        return "\n".join(result)

def main():
    parser = argparse.ArgumentParser(description="Inference pipeline for Ossetic Morphological Tagging and Lemmatization.")
    parser.add_argument("--text", type=str, default=None, help="Input Ossetic sentence.")
    parser.add_argument("--input_file", type=str, default=None, help="Input .txt file.")
    parser.add_argument("--output_file", type=str, default="output.conllu", help="Output CoNLL-U file.")
    parser.add_argument("--morph_model", type=str, default="ania3000/ossbert-morph-v2-1")
    parser.add_argument("--lemm_model", type=str, default="ania3000/ossbert-lemm-v2-1")
    parser.add_argument("--segmenter_path", type=str, default="word_segmenter_bundle-2-1.pth")
    
    args = parser.parse_args()
    pipeline = OsseticPipeline(args.morph_model, args.lemm_model, args.segmenter_path)

    if args.text:
        print(f"\n# text = {args.text}\n{pipeline.analyze_text(args.text)}\n")
    elif args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as fin, open(args.output_file, "w", encoding="utf-8") as fout:
            for cnt, line in enumerate(fin, start=1):
                sentence = line.strip()
                if not sentence: continue
                parsed = pipeline.analyze_text(sentence)
                fout.write(f"# sent_id = {cnt}\n# text = {sentence}\n{parsed}\n\n")
        print(f"Saved predictions to {args.output_file}")

if __name__ == "__main__":
    main()
