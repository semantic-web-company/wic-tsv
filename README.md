# Words-in-Context: Target Sense Verification Dataset

We present WiC-TSV, a new multi-domain evaluation benchmark for Word Sense Disambiguation. More specifically, we introduce a framework for Target Sense Verification of Words in Context which grounds its uniqueness in the formulation as a binary classification task thus being independent of external sense inventories, and the coverage of various domains. This makes the dataset highly flexible for the evaluation of a diverse set of models and systems in and across domains. WiC-TSV provides three different evaluation settings, depending on the input signals provided to the model.

For further details see [WiC-TSV arXiv paper](https://arxiv.org/abs/2004.15016).

See also our [challenge at codalab](https://competitions.codalab.org/competitions/23683).

For the latest results check out [WiC-TSV at paperswithcode](https://paperswithcode.com/dataset/wic-tsv).

### Structure

[data](./data) Contains the WiC-TSV dataset. The default format is in subfolder `Training`, `Development` and `Test`.
Other serializations are also stored in this folder.

[dataset_creation](./dataset_creation) Contains code related to the creation and cleaning of the dataset
in [data](./data).

[model_evaluation](./model_evaluation) Contains code  to read the dataset as torch dataset as well as additional scripts to prepare for the
evaluation and to get the final scores from the predictions.


### Cite

```
@ARTICLE{breit2021wictsv,
       author = {{Breit}, Anna and {Revenko}, Artem and {Rezaee}, Kiamehr and {Taher Pilehvar}, Mohammad and {Camacho-Collados}, Jose},
        title = "{WiC-TSV: An Evaluation Benchmark for Target Sense Verification of Words in Context}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computation and Language},
         year = 2020,
        month = apr,
          eid = {arXiv:2004.15016},
        pages = {arXiv:2004.15016},
archivePrefix = {arXiv},
       eprint = {2004.15016},
 primaryClass = {cs.CL},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020arXiv200415016B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
