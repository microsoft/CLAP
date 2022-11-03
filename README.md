# CLAP

CLAP (Contrastive Language-Audio Pretraining) is a neural network model that learns acoustic concepts from natural language supervision. It achieves SoTA in “Zero-Shot” classification, Audio-Text & Text-Audio Retrieval, and in some datasets when finetuned.

<img width="832" alt="clap_diagram_v3" src="https://user-images.githubusercontent.com/26778834/199842089-39ef6a2e-8abb-4338-bdfe-680abab70f53.png">

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

## Usage

```python
from CLAP_API import CLAP 

clap_model = CLAP("<PATH TO WEIGHTS>", use_cuda=False)

audio_files = ["audio_file1.wav", "audio_file2.wav"]
class_labels = ["label1", "label2", "label3", "label4"]

# get audio embeddings for downstream applications 
audio_embeddings = clap_model.get_audio_embeddings(audio_files)

# get text embeddings for downstream applications 
text_embeddings = clap_model.get_text_embeddings(class_labels)
```

## Examples

### Zero-Shot Prediction

The code below performs zero-shot prediction using CLAP. This example takes an audio from the [ESC50 dataset](https://github.com/karolpiczak/ESC-50), and predicts the most likely labels among the 50 textual labels from the dataset.

```python
from CLAP_API import CLAP
from esc50 import ESC50
import time
import torch.nn.functional as F

# Load CLAP
weights_path = # Add weight path here
model = CLAP(weights_path, use_cuda=False)

# Load dataset
dataset = ESC50(root='data', download=False)
x, target = dataset[1000]
y = dataset.classes

# Add prompt
prompt = 'this is a sound of '
y = dataset.classes
y_queries = [prompt + x for x in y]

# Compute embeddings and similarity matrix
text_embeddings = model.get_text_embeddings(y)
audio_embeddings = model.get_audio_embeddings(x, resample=True)
similarity = model.compute_similarity(audio_embeddings, text_embeddings)
similarity = F.softmax(similarity, dim=1)
values, indices = similarity[0].topk(5)

# Print the result
print("Ground Truth: {}".format(target))
print("Top predictions:\n")
for value, index in zip(values, indices):
    print(f"{y[index]:>16s}: {100 * value.item():.2f}%")
```

The output (the exact numbers may vary):

```
Ground Truth: coughing
Top predictions:

        coughing: 86.34%
        sneezing: 9.30%
drinking sipping: 1.31%
        laughing: 1.20%
  glass breaking: 0.81%
```

Note that this example uses the `get_text_embeddings()` and `get_audio_embeddings()` methods that return the encoded features of given inputs.

### Zero-Shot Evaluation

The code below performs zero-shot evaluation using CLAP to compute performance on [ESC50 dataset](https://github.com/karolpiczak/ESC-50) dataset.

```python
from CLAP_API import CLAP
from esc50 import ESC50
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Load CLAP
weights_path = # Add weight path
clap_model = CLAP(weights_path, use_cuda=False)

# Load dataset
dataset = ESC50(root='data', download=False)
prompt = 'this is a sound of '
Y = [prompt + x for x in dataset.classes]

# Computing text embeddings
text_embeddings = clap_model.get_text_embeddings(Y)
# Computing audio embeddings
y_preds, y_labels = [], []
for i in tqdm(range(len(dataset))):
    x, _, one_hot_target = dataset.__getitem__(i)

    audio_embedding = clap_model.get_audio_embeddings([x], resample=True)
    similarity = clap_model.compute_similarity(audio_embedding, text_embeddings)
    y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
    y_preds.append(y_pred)
    y_labels.append(one_hot_target.detach().cpu().numpy())

y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)
acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
print('ESC50 Accuracy {}'.format(acc))
```
The output:

```
ESC50 Accuracy: 82.6%
```

### Extract embeddings
- Extract text embeddings
```python

text_embeddings = clap_model.get_text_embeddings(class_labels: List[str])
text_embeddings = text_embeddings/torch.norm(text_embeddings, dim=-1, keepdim=True)
```

- Extract audio embeddings
```python

audio_embeddings = clap_model.get_audio_embeddings(file_paths: List[str])
audio_embeddings = audio_embeddings/torch.norm(audio_embeddings, dim=-1, keepdim=True)
```

- Compute similarity 
```python
# For using the below function, DO NOT normalize the text and audio embeddings
sim = clap_model.compute_similarity(audio_embeddings, text_embeddings)
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
