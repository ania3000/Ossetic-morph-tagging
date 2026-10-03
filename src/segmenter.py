import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMCharTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

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
