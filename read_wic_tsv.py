from pathlib import Path

from transformers import BertTokenizer

from model_evaluation import data_processors as dp
from model_evaluation import wictsv_dataset as wd


if __name__ == '__main__':
    train_folder = Path('data/Training')
    contexts, target_inds, hypernyms, definitions, labels = dp.read_wic_tsv_test(wic_tsv_folder=train_folder)
    wt_ds = wd.WiCTSVDataset(
        contexts=contexts,
        target_inds=target_inds,
        hypernyms=hypernyms,
        definitions=definitions,
        labels=labels,
        tokenizer=BertTokenizer.from_pretrained('bert-base-cased')
    )