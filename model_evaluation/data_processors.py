import csv
from pathlib import Path


def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r") as f:
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
        t_ind_start = t_ind if type(t_ind) == int else t_ind[0]
        if not cxt.split(' ')[t_ind_start].lower().startswith(tgt.split(' ')[0][:-1].lower()):
            assert False, (tgt.lower(), t_ind_start, cxt.split(' '), cxt.split(' ')[t_ind_start].lower())
    return contexts, target_inds, hypernyms, definitions, labels
