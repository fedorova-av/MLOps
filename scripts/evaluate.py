"""
DVC stage: evaluate
  - Загружает model_v2
  - Оценивает на Test_2
  - Пишет метрики в metrics/test_metrics.json
"""

import torch, torch.nn as nn, yaml, json
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score
)
from PIL import Image
import pandas as pd
from pathlib import Path
from tqdm import tqdm

with open('params.yaml') as f:
    params = yaml.safe_load(f)

BATCH_SIZE = params['evaluate']['batch_size']
DATA_ROOT  = 'ai-vs-human-generated-dataset-hw'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data     = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        img_path = self.root_dir / self.data.iloc[idx]['file_name']
        if not img_path.exists():
            img_path = Path(str(img_path).replace('train_data', 'test_data'))
        img = Image.open(img_path).convert('RGB')
        lbl = self.data.iloc[idx]['label']
        return (self.transform(img) if self.transform else img), lbl

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

test2  = ImageDataset(f'{DATA_ROOT}/Test_2/test.csv', f'{DATA_ROOT}/Test_2', transform)
loader = DataLoader(test2, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('models/model_v2.pth', map_location=device))
model = model.to(device)
model.eval()

criterion = nn.CrossEntropyLoss()
running_loss, all_preds, all_labels = 0.0, [], []

with torch.no_grad():
    for images, labels in tqdm(loader, desc='Evaluating v2 on Test_2'):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        running_loss += criterion(outputs, labels).item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

results = {
    'loss': running_loss / len(loader.dataset),
    'accuracy': accuracy_score(all_labels, all_preds),
    'f1': f1_score(all_labels, all_preds, average='weighted'),
    'precision': precision_score(all_labels, all_preds, average='weighted'),
    'recall': recall_score(all_labels, all_preds, average='weighted'),
}

writer = SummaryWriter(log_dir='my_logs/evaluate_v2')
for k, v in results.items():
    writer.add_scalar(f'Test_v2/{k}', v, 0)
writer.close()

Path('metrics').mkdir(exist_ok=True)
with open('metrics/test_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Evaluation results (v2 on Test_2):')
for k, v in results.items():
    print(f'  {k}: {v:.4f}')
