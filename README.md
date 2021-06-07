# Words-in-Context: Target Sense Verification Dataset

We present WiC-TSV, a new multi-domain evaluation benchmark for Word Sense Disambiguation. More specifically, we introduce a framework for Target Sense Verification of Words in Context which grounds its uniqueness in the formulation as a binary classification task thus being independent of external sense inventories, and the coverage of various domains. This makes the dataset highly flexible for the evaluation of a diverse set of models and systems in and across domains. WiC-TSV provides three different evaluation settings, depending on the input signals provided to the model.

To try out the dataset take these [10 interactive examples](https://www.surveymonkey.com/r/LHYWXPV).

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

[HyperBert](./HyperBert) Contains a few model identical to the ones used in the evaluation, extended with some ideas from the participants of the challenges, in particular, the paper [Vandenbussche, P. Y., Scerri, T., & Daniel Jr, R. (2021, January). Word Sense Disambiguation with Transformer Models. In Proceedings of the 6th Workshop on Semantic Deep Learning (SemDeep-6) (pp. 7-12).](https://www.aclweb.org/anthology/2021.semdeep-1.pdf) and [Moreno, J. G., Pontes, E. L., & Dias, G. (2021, January). CTLR@ WiC-TSV: Target Sense Verification using Marked Inputs andPre-trained Models. In Proceedings of the 6th Workshop on Semantic Deep Learning (SemDeep-6) (pp. 1-6).](https://www.aclweb.org/anthology/2021.semdeep-1.1.pdf).

## HyperBert Models

We have 2 different models:

* [HyperBert3](./HyperBert/HyperBert3.py) takes 3 different token representations: `[CLS]`, target token and sense identifiers. For the later 2 the representations are average across the tokens.
* [HyperBertCLS](./HyperBert/HyperBertCLS.py) takes only the representation of `[CLS]`.

*Remark*: **representations of a token** is the resulting output vector from the last layer of the respective language model -- sometimes also called **embedding**.

Both files can be run as scripts, for example, 
```bash
cd HyperBert
python3 HyperBert3 --dataset_path ../data --model_output_path ./eval --model_name bert-base-uncased
```
The parameters are optional, for defaults check the respective files.


### Cite

```
@inproceedings{breit-etal-2021-wic,
    title = "{WiC-TSV}: {A}n Evaluation Benchmark for Target Sense Verification of Words in Context",
    author = "Breit, Anna  and
      Revenko, Artem  and
      Rezaee, Kiamehr  and
      Pilehvar, Mohammad Taher  and
      Camacho-Collados, Jose",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.eacl-main.140",
    pages = "1635--1645",
    abstract = "We present WiC-TSV, a new multi-domain evaluation benchmark for Word Sense Disambiguation. More specifically, we introduce a framework for Target Sense Verification of Words in Context which grounds its uniqueness in the formulation as binary classification task thus being independent of external sense inventories, and the coverage of various domains. This makes the dataset highly flexible for the evaluation of a diverse set of models and systems in and across domains. WiC-TSV provides three different evaluation settings, depending on the input signals provided to the model. We set baseline performance on the dataset using state-of-the-art language models. Experimental results show that even though these models can perform decently on the task, there remains a gap between machine and human performance, especially in out-of-domain settings. WiC-TSV data is available at https://competitions.codalab.org/competitions/23683.",
}
```

### Acknowledgement

This work is supported has been supported by the European Union’s Horizon 2020 project Prêt-à-LLOD (grantagrement No 825182).