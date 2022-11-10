# CLAP

CLAP (Contrastive Language-Audio Pretraining) is a neural network model that learns acoustic concepts from natural language supervision. It achieves SoTA in “Zero-Shot” classification, Audio-Text & Text-Audio Retrieval, and in some datasets when finetuned.

<img width="832" alt="clap_diagram_v3" src="https://user-images.githubusercontent.com/26778834/199842089-39ef6a2e-8abb-4338-bdfe-680abab70f53.png">

## Setup

You are required to just install the dependencies: `pip install -r requirements.txt` using Python 3 to get started.

If you have [conda](https://www.anaconda.com) installed, you can run the following: 

```shell
git clone https://github.com/microsoft/CLAP.git && \
cd CLAP && \
conda create -n clap python=3.8 && \
conda activate clap && \
pip install -r requirements.txt
```

## CLAP weights
Request CLAP weights by filling this form: [link](https://forms.office.com/r/ULb4k9GL1F)


## Usage

Please take a look at `src/examples` for usage examples. 

- Load model
```python
from src import CLAP 

clap_model = CLAP("<PATH TO WEIGHTS>", use_cuda=False)
```

- Extract text embeddings
```python
text_embeddings = clap_model.get_text_embeddings(class_labels: List[str])
```

- Extract audio embeddings
```python
audio_embeddings = clap_model.get_audio_embeddings(file_paths: List[str])
```

- Compute similarity 
```python
sim = clap_model.compute_similarity(audio_embeddings, text_embeddings)
```

## Examples
To run zero-shot evaluation on the ESC50 dataset or a single audio file from ESC50, check `CLAP\src\`. For zero-shot evaluation on the ESC50 dataset:
```bash
> cd src && python zero_shot_classification.py
```
Output
```bash
ESC50 Accuracy: 82.6%
```

## Citation
https://arxiv.org/pdf/2206.04769.pdf
```
@article{elizalde2022clap,
  title={Clap: Learning audio concepts from natural language supervision},
  author={Elizalde, Benjamin and Deshmukh, Soham and Ismail, Mahmoud Al and Wang, Huaming},
  journal={arXiv preprint arXiv:2206.04769},
  year={2022}
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
