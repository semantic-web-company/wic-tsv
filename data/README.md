## WiC-TSV: the Word-in-Context Dataset for Target Sense Verification

For more information, please visit https://competitions.codalab.org/competitions/23683

Data is split in training, development and test. Each split contains four files.

* train/dev/test_examples.txt tab separated file including the target word and their contexts columns: target word |
  target index (word position) | context

* train/dev/test_definitions.txt file including the definitions line number corresponds to line number of example in
  train/dev/test_examples.txt

* train/dev/test_hypernyms.txt file including the hypernym/s, tab-separated hypernyms consisting of more than one token
  are split by underscore line number corresponds to line number of example in train/dev/test_examples.txt

* train/dev_labels.txt file including the gold labels (T-True or F-False)
  line number corresponds to line number of example in train/dev_examples.txt

NOTE: Test set does not include labels as they are kept secret, you can submit your results on CodaLab.
