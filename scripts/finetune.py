"""
DVC stage: finetune
  - Скачивает model_v1 из S3
  - Дообучает на Train_2
  - Логирует в TensorBoard
  - Сохраняет model_v2 локально и в S3
  - Пишет метрики в metrics/finetune_metrics.json
"""

import torch, torch.nn as nn, torch.optim as optim, yaml, json, boto3
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
from botocore.client import Config
from sklearn.metrics import f1_score, accuracy_score
from PIL import Image
import pandas as pd
from pathlib import Path
from tqdm import tqdm

with open('params.yaml') as f:
    params = yaml.safe_load(f)

p = params['finetune']
LR         = p['lr']
NUM_EPOCHS = p['num_epochs']
BATCH_SIZE = p['batch_size']
DATA_ROOT  = 'ai-vs-human-generated-dataset-hw'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

s3 = boto3.client(
    's3',
    endpoint_url=p['s3_endpoint'],
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin',
    config=Config(signature_version='s3v4'),
    region_name='us-east-1'
)
Path('models').mkdir(exist_ok=True)
s3.download_file(p['s3_bucket'], p['s3_key_v1'], 'models/model_v1.pth')
print('Downloaded model_v1.pth from S3')

class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data     = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
    def __len__(self):  return len(self.data)
    def __getitem__(self, idx):
        img  = Image.open(self.root_dir / self.data.iloc[idx]['file_name']).convert('RGB')
        lbl  = self.data.iloc[idx]['label']
        return (self.transform(img) if self.transform else img), lbl

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train2 = ImageDataset(f'{DATA_ROOT}/Train_2/train.csv', f'{DATA_ROOT}/Train_2', transform)
loader = DataLoader(train2, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('models/model_v1.pth', map_location=device))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

writer = SummaryWriter(log_dir='my_logs/finetune_v2')
writer.add_text('Hyperparameters/finetune', str(p), global_step=0)

metrics_log = {}
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss, all_preds, all_labels = 0.0, [], []

    for images, labels in tqdm(loader, desc=f'Finetune Epoch {epoch+1}'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    ep_loss = running_loss / len(loader.dataset)
    ep_acc = accuracy_score(all_labels, all_preds)
    ep_f1 = f1_score(all_labels, all_preds, average='weighted')

    writer.add_scalar('Loss/finetune', ep_loss, epoch)
    writer.add_scalar('Accuracy/finetune', ep_acc, epoch)
    writer.add_scalar('F1/finetune', ep_f1, epoch)

    print(f'Epoch {epoch+1}: loss={ep_loss:.4f}  acc={ep_acc:.4f}  f1={ep_f1:.4f}')
    metrics_log = {'loss': ep_loss, 'accuracy': ep_acc, 'f1': ep_f1}

writer.close()

torch.save(model.state_dict(), 'models/model_v2.pth')
s3.upload_file('models/model_v2.pth', p['s3_bucket'], p['s3_key_v2'])
print('Uploaded model_v2.pth to S3')

Path('metrics').mkdir(exist_ok=True)
with open('metrics/finetune_metrics.json', 'w') as f:
    json.dump(metrics_log, f, indent=2)
print('Finetune metrics saved')
