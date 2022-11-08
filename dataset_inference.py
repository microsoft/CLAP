from CLAP_API import CLAP
from esc50_dataset import ESC50
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Load CLAP
weights_path = 'C:\\Users\\sdeshmukh\\Desktop\\CLAP_package\\model\\new\\best.pth' # Add weight path here
clap_model = CLAP(weights_path, use_cuda=False)

# Load dataset
dataset = ESC50(root='data', download=True)
prompt = 'this is a sound of '
Y = [prompt + x for x in dataset.classes]

# Computing text embeddings
text_embeddings = clap_model.get_text_embeddings(Y)

# Computing audio embeddings
y_preds, y_labels = [], []
for i in tqdm(range(len(dataset))):
    x, _, one_hot_target = dataset.__getitem__(i)
    audio_embeddings = clap_model.get_audio_embeddings([x], resample=True)
    similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)
    y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
    y_preds.append(y_pred)
    y_labels.append(one_hot_target.detach().cpu().numpy())

y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)
acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
print('ESC50 Accuracy {}'.format(acc))