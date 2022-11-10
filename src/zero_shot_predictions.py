"""
This is an example using CLAP for zero-shot 
        inference using ESC50 (https://github.com/karolpiczak/ESC-50).
"""

from CLAPWrapper import CLAPWrapper
from esc50_dataset import ESC50
import torch.nn.functional as F

# Load ESC50 dataset
dataset = ESC50(root="data_path", download=True) # set download=True when dataset is not downloaded
audio_file, target, one_hot_target = dataset[1000]
audio_file = [audio_file]
prompt = 'this is a sound of '
y = [prompt + x for x in dataset.classes]

# Load and initialize CLAP
weights_path = "weights_path"

# Setting use_cuda = True will load the model on a GPU using CUDA
clap_model = CLAPWrapper(weights_path, use_cuda=False)

# compute text embeddings from natural text 
text_embeddings = clap_model.get_text_embeddings(y)

# compute the audio embeddings from an audio file 
audio_embeddings = clap_model.get_audio_embeddings(audio_file, resample=True)

# compute the similarity between audio_embeddings and text_embeddings
similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)

similarity = F.softmax(similarity, dim=1)
values, indices = similarity[0].topk(5)

# view the results
print("Ground Truth: {}".format(target))
print("Top predictions:\n")
for value, index in zip(values, indices):
    print(f"{dataset.classes[index]:>16s}: {100 * value.item():.2f}%")

"""
The output (the exact numbers may vary):

Ground Truth: coughing
Top predictions:

        coughing: 86.34%
        sneezing: 9.30%
drinking sipping: 1.31%
        laughing: 1.20%
  glass breaking: 0.81%
"""