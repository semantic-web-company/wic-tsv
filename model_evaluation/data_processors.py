import csv
from pathlib import Path

import pandas as pd


def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines


def read_wic_tsv(wic_tsv_folder: Path,
                 tgt_column=0,
                 tgt_ind_column=1,
                 cxt_column=2):
    targets = []
    contexts = []
    target_inds = []
    examples_path = next(wic_tsv_folder.glob('*_examples.txt'))
    for line in _read_tsv(examples_path):
        try:
            target_inds.append(int(line[tgt_ind_column].strip()))
        except ValueError:
            target_inds.append((int(line[tgt_ind_column].strip().split("-")[0]),
                                int(line[tgt_ind_column].strip().split("-")[-1]))
                               )
        contexts.append(line[cxt_column].strip())
        targets.append(line[tgt_column].strip())

    hypernyms = []
    hypernyms_path = next(wic_tsv_folder.glob('*_hypernyms.txt'))
    for line in _read_tsv(hypernyms_path):
        hypernyms.append([hypernym.replace('_', ' ').strip() for hypernym in line])

    defs_path = next(wic_tsv_folder.glob('*_definitions.txt'))
    definitions = [definition[0] for definition in _read_tsv(defs_path)]

    try:
        labels_path = next(wic_tsv_folder.glob('*_labels.txt'))
        labels = [int(x[0].strip() == 'T') for x in _read_tsv(labels_path)]
    except Exception as e:
        print(e)
        labels = None
    assert len(contexts) == len(hypernyms) == len(definitions), (len(contexts), len(hypernyms), len(definitions))
    for cxt, t_ind, tgt in zip(contexts, target_inds, targets):
        t_ind_start = t_ind if isinstance(t_ind, int) else t_ind[0]
        if not cxt.split(' ')[t_ind_start].lower().startswith(tgt.split(' ')[0][:-1].lower()):
            assert False, (tgt.lower(), t_ind_start, cxt.split(' '), cxt.split(' ')[t_ind_start].lower())
    return contexts, target_inds, hypernyms, definitions, labels


def read_conll(file_path: Path,
               token_col: int,
               ne_col: int,
               delimiter: str = " ",
               quote_char: str = "\\",
               doc_split_str='-DOCSTART- -X- -X- -X- O\n',
               sent_split_str="\n\n"
               ):
    df = pd.read_csv(file_path,
                     delimiter=delimiter,
                     quotechar=quote_char,
                     keep_default_na=False,
                     header=None)
    df.index = df[token_col]
    df = df[token_col, ne_col]
    df.columns = ["token", "ne"]

    # remove lines indicating a new document
    df = df.drop(doc_split_str.split(delimiter)[0])
    df = df.reset_index(drop=True)

    unique_nes = df['ne'].unique()

    # add sentence and doc ids
    with open(file_path, "r") as f:
        doc_ids, sent_ids = get_conll_doc_and_sent_ids(f.read(), doc_split_str, sent_split_str)
    assert (df.shape[0] == len(doc_ids) and df.shape[0] == len(
        sent_ids)), "Number of doc-ids/sent-ids does not match with number of words "
    df['sent'] = sent_ids
    # df['doc'] = doc_ids

    sent_grouped = df.groupby("sent")
    tokens = sent_grouped.apply(lambda s: [x for x in s["word"].values.tolist()])
    nes = sent_grouped.apply(lambda s: [x for x in s["ne"].values.tolist()])

    return tokens, nes, unique_nes


def get_conll_doc_and_sent_ids(file_content: str, doc_split_str='-DOCSTART- -X- -X- -X- O\n', sent_split_str="\n\n"):
    raw_docs = file_content.split(doc_split_str)
    raw_sens = [doc.split(sent_split_str) for doc in raw_docs if doc is not '']
    doc_ids = []
    sent_ids = []
    current_sent_id = 0
    for doc_i in range(len(raw_sens)):
        for sent_i in range(len(raw_sens[doc_i])):
            num_lines = len([line for line in raw_sens[doc_i][sent_i].split('\n') if line is not ""])
            doc_ids.extend([doc_i] * num_lines)
            sent_ids.extend([current_sent_id] * num_lines)
            current_sent_id += 1
    return doc_ids, sent_ids
