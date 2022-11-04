from CLAP_API import CLAP
from esc50 import ESC50
import time
import torch.nn.functional as F

start_time = time.time()
weights_path = 'C:\\Users\\sdeshmukh\\Desktop\\CLAP_package\\model\\new\\best.pth'
clap_model = CLAP(weights_path, use_cuda=False)
print("Finished loading CLAP. Total time: {}".format(time.time() - start_time))

esc50_dataset = ESC50(root='data', download=False)
x, target = esc50_dataset[1000]
x = [x]
y = esc50_dataset.classes

print('Computing text embeddings')
text_embeddings = clap_model.get_text_embeddings(y)
print('Computing audio embeddings')
audio_embeddings = clap_model.get_audio_embeddings(x, resample=True)
similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)

similarity = F.softmax(similarity, dim=1)
values, indices = similarity[0].topk(5)
# Print the result
print("Ground Truth: {}".format(target))
print("Top predictions:\n")
for value, index in zip(values, indices):
    print(f"{y[index]:>16s}: {100 * value.item():.2f}%")