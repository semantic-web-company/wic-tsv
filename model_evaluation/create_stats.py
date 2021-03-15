import os
from pathlib import Path

import numpy as np


def remove_zero_entries(d: dict):
    new_d = {k: v for k, v in d.items() if v != 0}
    return new_d


class DatasetSplit:
    def __init__(self, data_folder: Path,
                 examples_file_name: str,
                 definitions_file_name: str,
                 hypernyms_file_name: str,
                 labels_file_name: str,
                 domains_file_name: str = None):
        self.domain_dict = {
            'wnt': ["0"],
            'msh': ["1"],
            'ctl': ["2"],
            'cps': ["3"],
            'domains': ["1", "2", "3"]
        }

        self.examples = [line.split('\t') for line in
                         (data_folder / examples_file_name).read_text().strip('\n').split('\n')]
        self.defs = [line for line in
                     (data_folder / definitions_file_name).read_text().strip('\n').split('\n')]
        self.hyps = [line.split('\t') for line in
                     (data_folder / hypernyms_file_name).read_text().strip('\n').split('\n')]
        self.labels = [line for line in
                       (data_folder / labels_file_name).read_text().strip('\n').split('\n')]
        if domains_file_name is not None:
            self.domains = [line for line in
                            (data_folder / domains_file_name).read_text().strip('\n').split('\n')]
        else:
            self.domains = None

        self.targets = ([l[0] for l in self.examples])


    def num_total_examples(self, domain=None):
        if domain is not None:
            return len(list(np.where(np.isin(self.domains, self.domain_dict[domain]))[0]))
        return len(self.examples)


    def num_unique_senses(self, domain=None):
        if domain is not None:
            indices = list(np.where(np.isin(self.domains, self.domain_dict[domain]))[0])
            return len(set(list(np.array(self.defs)[indices])))
        return len(set(self.defs))


    def num_unique_targets(self, domain):
        if domain is not None:
            indices = list(np.where(np.isin(self.domains, self.domain_dict[domain]))[0])
            return len(set(list(np.array(self.targets)[indices])))
        return len(set(self.targets))


    def examples_per_sense(self, domain=None):
        """
        creates a list of number of examples per sense provided in the dataset
        outcome: ordered list by numbers of used examples per sense
        example: {1:10, 2:5, 4:15} means that for one sense 10 examples have been provided,
        for 2 senses 5 examples have been provided, for 4 senses have been provided and so on
        """
        if domain is not None:
            indices = list(np.where(np.isin(self.domains, self.domain_dict[domain]))[0])
            domain_defs = list(np.array(self.defs)[indices])
            examples_per_sense = [domain_defs.count(x) for x in set(domain_defs)]
        else:
            examples_per_sense = [self.defs.count(x) for x in set(self.defs)]
        count_examples_per_sense = remove_zero_entries({i: examples_per_sense.count(i) for i in range(1, 100)})
        return count_examples_per_sense


    def examples_per_target(self, domain: list = None):
        if domain is not None:
            indices = list(np.where(np.isin(self.domains, self.domain_dict[domain]))[0])
            domain_tgts = list(np.array(self.targets)[indices])
            nof_examples_per_target = [domain_tgts.count(x) for x in set(domain_tgts)]
        else:
            nof_examples_per_target = [self.targets.count(x) for x in set(self.targets)]
        count_examples_per_target = remove_zero_entries({i: nof_examples_per_target.count(i) for i in range(1, 100)})
        return count_examples_per_target


    def percent_positive_examples(self, domain=None):
        if domain is not None:
            indices = list(np.where(np.isin(self.domains, self.domain_dict[domain]))[0])
            return list(np.array(self.labels)[indices]).count('T') / len(indices)
        return self.labels.count('T') / len(self.labels)


    def target_overlap(self, overlap_dataset, second_dataset=None, domain=None):
        if domain is not None:
            indices = list(np.where(np.isin(self.domains, self.domain_dict[domain]))[0])
        else:
            indices = list(range(len(self.targets)))
        if second_dataset is not None:
            unique_targets_a = set(np.array(self.targets)[indices]) | set(second_dataset.targets)
        else:
            unique_targets_a = set(np.array(self.targets)[indices])
        overlap_targets = unique_targets_a & set(overlap_dataset.targets)
        return len(overlap_targets) / len(overlap_dataset.targets)


    def sense_overlap(self, overlap_dataset, second_dataset=None, domain=None):
        if domain is not None:
            indices = list(np.where(np.isin(self.domains, self.domain_dict[domain]))[0])
        else:
            indices = list(range(len(self.defs)))
        if second_dataset is not None:
            unique_defs_a = set(np.array(self.defs)[indices]) | set(second_dataset.defs)
        else:
            unique_defs_a = set(np.array(self.defs)[indices])
        overlap_defs = unique_defs_a & set(overlap_dataset.defs)
        return len(overlap_defs) / len(overlap_dataset.defs)


    def instance_overlap(self, overlap_dataset, second_dataset=None):
        instances_self = set([e + d for e, d in zip(self.examples, self.defs)])
        instances_overlap = set([e + d for e, d in zip(overlap_dataset.examples, overlap_dataset.defs)])
        if second_dataset is not None:
            instances_second = set([e + d for e, d in zip(second_dataset.examples, second_dataset.defs)])
            unique_instances_a = instances_self | instances_second
        else:
            unique_instances_a = instances_self
        overlap_defs = unique_instances_a & set(instances_overlap)
        return len(overlap_defs) / len(instances_overlap)


def get_pos_label_percent(folder: str, prefix: str, label_indices: list = None):
    labels = open(os.path.join(folder, prefix + 'labels.txt')).read().split("\n")
    if label_indices is not None:
        labels = list(np.array(labels)[label_indices])
    return labels.count('T') / len(labels)


def get_unique_senses(folder: str, prefix: str, definition_indices: list = None):
    definitions = open(os.path.join(folder, prefix + 'definitions.txt')).read().split("\n")
    if definition_indices is not None:
        definitions = list(np.array(definitions)[definition_indices])
    return len(set(definitions))


def get_total_number(folder: str, prefix: str, example_indices: list = None):
    examples = open(os.path.join(folder, prefix + 'definitions.txt')).read().split("\n")
    if example_indices is not None:
        examples = list(np.array(examples)[example_indices])
    return len(examples)


def print_stats(dataset: DatasetSplit, domain=None):
    print('total number', dataset.num_total_examples(domain=domain))
    print('number unique senses', dataset.num_unique_senses(domain=domain))
    print('number unique targets', dataset.num_unique_targets(domain=domain))
    print('percent positive labels', dataset.percent_positive_examples(domain=domain))


if __name__ == '__main__':
    location = Path(__file__).parent.parent
    train_folder = location / 'data' / 'Training'
    dev_folder = location / 'data' / 'Development'
    test_folder = location / 'data' / 'Test'

    train_folder = location / 'data' / 'release_v1' / 'Training'
    dev_folder = location / 'data' / 'release_v1' / 'Development'
    test_folder = location / 'data' / 'release_v1' / 'Test'

    train_split = DatasetSplit(data_folder=Path(train_folder),
                               examples_file_name='train_examples.txt',
                               definitions_file_name='train_definitions.txt',
                               hypernyms_file_name='train_hypernyms.txt',
                               labels_file_name='train_labels.txt')
    dev_split = DatasetSplit(data_folder=Path(dev_folder),
                             examples_file_name='dev_examples.txt',
                             definitions_file_name='dev_definitions.txt',
                             hypernyms_file_name='dev_hypernyms.txt',
                             labels_file_name='dev_labels.txt')
    # test_split = DatasetSplit(data_folder=Path(test_folder),
    #                          examples_file_name='test_examples.txt',
    #                          definitions_file_name='test_definitions.txt',
    #                          hypernyms_file_name='test_hypernyms.txt',
    #                          labels_file_name='test_labels.txt',
    #                          domains_file_name='test_domains.txt')

    print(train_split.target_overlap(dev_split))
    # print(train_split.target_overlap(test_split))
    # print(train_split.target_overlap(test_split, dev_split))
    print()
    # print(test_split.target_overlap(train_split))
    # print(test_split.target_overlap(train_split, domain='wnt'))
    # print(test_split.target_overlap(train_split, domain='msh'))
    # print(test_split.target_overlap(train_split, domain='ctl'))
    # print(test_split.target_overlap(train_split, domain='cps'))
    print()

    print(train_split.sense_overlap(dev_split))
    # print(train_split.sense_overlap(test_split))
    # print(train_split.sense_overlap(test_split, dev_split))
    print()

    # print(test_split.sense_overlap(train_split))
    # print(test_split.sense_overlap(train_split, domain='wnt'))
    # print(test_split.sense_overlap(train_split, domain='msh'))
    # print(test_split.sense_overlap(train_split, domain='ctl'))
    # print(test_split.sense_overlap(train_split, domain='cps'))
    print('\n##### train #####')
    print_stats(train_split)

    print('\n##### dev #####')
    print_stats(dev_split)

    # print('\n##### test #####')
    # print_stats(test_split)
    #
    # print('\n##### wnt')
    # print_stats(test_split, 'wnt')
    #
    # print('\n##### msh')
    # print_stats(test_split, 'msh')
    #
    # print('\n##### ctl')
    # print_stats(test_split, 'ctl')
    #
    # print('\n##### cps')
    # print_stats(test_split, 'cps')
    #
    # print('\n##### all domains')
    # print_stats(test_split, 'domains')
