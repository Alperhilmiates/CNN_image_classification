#%%
import torch
import torchvision.transforms as T
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from PIL import Image

# Define the dataset and data loader
data_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])
train_dataset = datasets.ImageFolder('train', transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# Load the ResNet-50 model
model = models.resnet50(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])  # Remove the last fully connected layer

# Define the contrastive learning loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x, y, labels):
        # Compute the cosine similarity between the embeddings
        similarity = nn.functional.cosine_similarity(x, y)
        # Compute the contrastive loss
        loss = torch.mean((1 - labels) * torch.square(similarity) +
                          labels * torch.square(torch.clamp(self.margin - similarity, min=0)))
        return loss

# Train the model using contrastive learning
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = ContrastiveLoss()
#%%
num_epochs = 10
for epoch in range(num_epochs):
    for images, _ in train_loader:
        # Augment the images and split them into two views
        views = [data_transform(image) for image in images]
        views = torch.stack(views)
        views = views.split(1, dim=0)
        # Forward pass through the model and compute the embeddings
        embeddings = [model(view) for view in views]
        # Compute the contrastive loss and update the model
        loss = criterion(embeddings[0], embeddings[1], torch.ones(images.size(0)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Extract the learned features and train a linear SVM
train_features = []
train_labels = []
with torch.no_grad():
    for images, labels in train_loader:
        features = model(images)
        features = features.reshape(features.size(0), -1)
        train_features.append(features)
        train_labels.append(labels)
train_features = torch.cat(train_features, dim=0)
train_labels = torch.cat(train_labels, dim=0)

# Define the SVM classifier
svm = LinearSVC()

# Train the SVM classifier on the extracted features
svm.fit(train_features, train_labels)

# Define the test dataset and data loader
test_dataset = datasets.ImageFolder('test', transform=data_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluate the SVM classifier on the test data
test_features = []
test_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        features = model(images)
        features = features.reshape(features.size(0), -1)
        test_features.append(features)
        test_labels.append(labels)
test_features = torch.cat(test_features, dim=0)
test_labels = torch.cat(test_labels, dim=0)

# Predict the labels of the test data using the trained SVM
test_predictions = svm.predict(test_features)

# Calculate the accuracy of the predictions
accuracy = accuracy_score(test_labels, test_predictions)
print(f"Test accuracy: {accuracy}")