from CLAP_API import CLAP
from esc50_dataset import ESC50
import time
import torch.nn.functional as F

# Load CLAP
weights_path = 'C:\\Users\\sdeshmukh\\Desktop\\CLAP_package\\model\\new\\best.pth' # Add weight path here
clap_model = CLAP(weights_path, use_cuda=False)

# Load dataset
dataset = ESC50(root='data', download=True) # set download=True when dataset is not downloaded
audio_file, target, one_hot_target = dataset[1000]
audio_file = [audio_file]
prompt = 'this is a sound of '
y = [prompt + x for x in dataset.classes]

print('Computing text embeddings')
text_embeddings = clap_model.get_text_embeddings(y)
print('Computing audio embeddings')
audio_embeddings = clap_model.get_audio_embeddings(audio_file, resample=True)
similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)

similarity = F.softmax(similarity, dim=1)
values, indices = similarity[0].topk(5)
# Print the result
print("Ground Truth: {}".format(target))
print("Top predictions:\n")
for value, index in zip(values, indices):
    print(f"{dataset.classes[index]:>16s}: {100 * value.item():.2f}%")