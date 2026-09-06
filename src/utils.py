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
