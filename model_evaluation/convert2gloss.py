from pathlib import Path

from model_evaluation.data_processors import read_wic_tsv
from model_evaluation.wictsv_dataset import WiCTSVDataset

base_path = Path(__file__).parent.parent / 'data'
wic_tsv_train = base_path / 'Training'
wic_tsv_dev = base_path / 'Development'
wic_tsv_test = base_path / 'Test'

if __name__ == "__main__":
    for data_path, ds_type in [(wic_tsv_train, 'train'), (wic_tsv_dev, 'dev'), (wic_tsv_test, 'test')]:
        data = read_wic_tsv(data_path)
        contexts, target_inds, hypernyms, definitions, labels = data
        single_data = zip(labels, contexts, definitions)
        with open(base_path / f'{ds_type}_gloss.csv', 'w') as f:
            f.write(f'ID\tLabel\tContext\tDefinition\n')
            for i, (l, cxt, def_) in enumerate(single_data):
                f.write(f'{i}\t{int(l)}\t{cxt}\t{def_}\n')

        marked_contexts, marked_target_inds = WiCTSVDataset.mark_target_in_context(contexts=contexts,
                                                                                   target_inds=target_inds,
                                                                                   focus_char='"')
        single_data = zip(labels, marked_contexts, definitions)
        with open(base_path / f'{ds_type}_gloss_ws.csv', 'w') as f:
            f.write(f'ID\tLabel\tContext\tDefinition\n')
            for i, (l, cxt, def_) in enumerate(single_data):
                f.write(f'{i}\t{int(l)}\t{cxt}\t{def_}\n')
