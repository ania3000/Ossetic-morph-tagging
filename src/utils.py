import re
from functools import reduce

try:
    from pyparadigm.paradigm import LcsSearcher, find_gap_positions, compute_paradigm
    HAS_PYPARADIGM = True
except ImportError:
    HAS_PYPARADIGM = False

SUPPL_DICT = {
    'дзы'+'1+йæ#1+дзы': 'йæ',
    'дзы'+'1+сæ#1+дзы': 'сæ',
    'сыл'+'1+cæ#1+сыл': 'cæ',
    'сын'+'1+cæ#1+сын': 'cæ',
    'ыл'+'1+йæ#1+ыл': 'йæ',
    'ын'+'1+йæ#1+ын': 'йæ',
}

def make_last_subtoken_mask(word_ids, has_cls=True, has_eos=True):
    mask = word_ids
    if has_cls:
        mask = mask[1:]
    if has_eos:
        mask = mask[:-1]
    is_last_word = [first != second for first, second in zip(mask[:-1], mask[1:])] + [True]
    if has_cls:
        is_last_word = [False] + is_last_word
    if has_eos:
        is_last_word.append(False)
    return is_last_word

def get_ossetic_label(lemma, word_form):
    """Генерирует редакционную метку парадигмы на основе LCS."""
    if not HAS_PYPARADIGM:
        raise ImportError("pyparadigm не установлен. Скомпилируйте C++ модуль pyparadigm.")
        
    searcher = LcsSearcher(
        gap=3, initial_gap=None, method='Hulden', 
        min_constant_count=0, remove_constant_variables=True
    )
    best_lcss = searcher.process_table([word_form, lemma])

    if not best_lcss or (not best_lcss[0][0] and all(not idx_list for idx_list in best_lcss[0][1])):
        return f"1+{lemma}#1+{word_form}"

    lcs_string, lcs_indexes = best_lcss[0]
    all_gaps = [set(find_gap_positions(word_indexes)) for word_indexes in lcs_indexes]
    unique_gaps = sorted(reduce(lambda x, y: x | y, all_gaps, set()))
    var_beginnings = [0] + unique_gaps + [len(lcs_string)]

    answer = compute_paradigm(
        table=[word_form, lemma],
        indexes=lcs_indexes,
        var_beginnings=var_beginnings
    )
    return f"{answer[1]}#{answer[0]}"

def restore_lemma(word_form, label):
    """Восстанавливает лемму из формы слова и редакционной метки."""
    try:
        lemma_rule, form_rule = label.split('#')
        form_parts = form_rule.split('+')

        regex_pattern = ""
        var_order = []
        for part in form_parts:
            if part.isdigit():
                regex_pattern += r"(.+)"
                var_order.append(int(part))
            else:
                regex_pattern += re.escape(part)

        match = re.match(f"^{regex_pattern}$", word_form) or re.match(f"^{regex_pattern}$", word_form.lower())
        if match:
            extracted_vars = {var_num: val for var_num, val in zip(var_order, match.groups())}
        else:
            return SUPPL_DICT.get(word_form + label, word_form)

        lemma_parts = lemma_rule.split('+')
        final_lemma_pieces = [
            extracted_vars.get(int(part), "") if part.isdigit() else part 
            for part in lemma_parts
        ]
        return "".join(final_lemma_pieces)

    except Exception:
        return word_form

def read_lemmatization_conllu(infile):
    """Парсит CoNLL-U для задачи лемматизации."""
    answer, sent, labels, lemmas = [], [], [], []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if line == "" or line.startswith("#"):
                if len(sent) > 0:
                    answer.append({"words": sent, "labels": labels, 'lemmas': lemmas})
                sent, labels, lemmas = [], [], []
                continue

            splitted = line.split("\t")
            if not splitted[0].isdigit() or len(splitted) < 4:
                continue

            wordform = re.sub("Ӕ", "Æ", re.sub("ӕ", "æ", splitted[1]))
            lemma = re.sub("Ӕ", "Æ", re.sub("ӕ", "æ", splitted[2]))

            tag = get_ossetic_label(lemma.lower(), wordform.lower())

            sent.append(wordform)
            labels.append(tag)
            lemmas.append(lemma)

    if len(sent) > 0:
        answer.append({"words": sent, "labels": labels, "lemmas": lemmas})
    return answer

def make_last_subtoken_mask(word_ids, has_cls=True, has_eos=True):
    """Создает маску для выбора последнего субтокена каждого слова."""
    mask = word_ids
    if has_cls:
        mask = mask[1:]
    if has_eos:
        mask = mask[:-1]
    is_last_word = [first != second for first, second in zip(mask[:-1], mask[1:])] + [True]
    if has_cls:
        is_last_word = [False] + is_last_word
    if has_eos:
        is_last_word.append(False)
    return is_last_word


def read_conllu(infile):
    """Парсит файлы формата CoNLL-U."""
    answer, sent, labels = [], [], []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                if sent:
                    answer.append({"words": sent, "labels": labels})
                sent, labels = [], []
                continue
            if line.startswith("#"):
                continue
            splitted = line.split("\t")
            if not splitted[0].isdigit() or len(splitted) < 4:
                continue
            
            tag = splitted[3] if splitted[5] == "_" else f"{splitted[3]},{splitted[5]}"
            sent.append(splitted[1])
            labels.append(tag)
            
    if sent:
        answer.append({"words": sent, "labels": labels})
    return answer

def get_all_tasks(infile_path):
    """Автоматически собирает список всех уникальных морфологических фичей из CoNLL-U."""
    all_tasks = {"POS"}
    with open(infile_path, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            splitted = line.split("\t")
            if len(splitted) >= 6 and splitted[5] != "_":
                for feat in splitted[5].split("|"):
                    key, _ = feat.split("=")
                    all_tasks.add(key)
    return sorted(list(all_tasks))

def read_mt_conllu(infile, task_names):
    """Парсит CoNLL-U в словарь мультитаск-меток."""
    answer = []
    with open(infile, "r", encoding="utf8") as fin:
        sent = []
        labels = {task: [] for task in task_names}
        raw_labels = []

        for line in fin:
            line = line.strip()
            if not line:
                if sent:
                    answer.append({
                        "words": sent, 
                        "labels": {k: v[:] for k, v in labels.items()},
                        "raw_labels": raw_labels[:]
                    })
                sent = []
                labels = {task: [] for task in task_names}
                raw_labels = []
                continue

            splitted = line.split("\t")
            if not splitted[0].isdigit() or len(splitted) < 6:
                continue

            sent.append(splitted[1])
            pos_tag, feats = splitted[3], splitted[5]
            
            # Сохраняем цельную метку для честного подсчета Full Tag Accuracy
            full_tag = pos_tag if feats == "_" else f"{pos_tag},{feats}"
            raw_labels.append(full_tag)

            labels["POS"].append(pos_tag)
            feats_dict = {}
            if feats != "_":
                for feat in feats.split("|"):
                    key, val = feat.split("=")
                    feats_dict[key] = val

            for task in task_names:
                if task != "POS":
                    labels[task].append(feats_dict.get(task, "None"))

        if sent:
            answer.append({
                "words": sent, 
                "labels": {k: v[:] for k, v in labels.items()},
                "raw_labels": raw_labels[:]
            })
    return answer
