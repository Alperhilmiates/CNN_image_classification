
#%% packages
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
os.getcwd()
from torchsummary import summary
# %% transform and load data
# set up image transforms
transform = transforms.Compose([transforms.Resize((100, 100)),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# set up train and test datasets
trainset = torchvision.datasets.ImageFolder(root='train', transform=transform)
testset = torchvision.datasets.ImageFolder(root='test', transform=transform)
# set up data loaders
batch_size = 4
trainloader = DataLoader(trainset, batch_size = batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size = batch_size, shuffle=True)

# %% visualize images
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images, nrow=2))

# %%
CLASSES = ['affenpinscher', 'akita', 'corgi']
NUM_CLASSES = len(CLASSES)

# TODO: set up model class
class ImageMulticlassClassificationNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3) # BS, C, H, W output size = (W-F+2P)/S + 1 = (100-3+2)/1 + 1 = 100
        self.pool = nn.MaxPool2d(2, 2) # BS, C, H, W output size = (W-F+2P)/S + 1 = (100-2+2)/2 + 1 = 50
        self.conv2 = nn.Conv2d(6, 16, 3) # BS, C, H, W output size = (W-F+2P)/S + 1 = (50-3+2)/1 + 1 = 50
        self.conv3 = nn.Conv2d(16, 32, 3) # BS, C, H, W output size = (W-F+2P)/S + 1 = (50-3+2)/1 + 1 = 50
        self.fc1 = nn.Linear(32 * 10 * 10, 128) # input channel = 32*12*12, output channel = 128
        self.fc2 = nn.Linear(128, 64) # input channel = 128, output channel = 64
        self.fc3 = nn.Linear(64, 16) # input channel = 64, output channel = 16
        self.fc4 = nn.Linear(16, NUM_CLASSES) # input channel = 16, output channel = NUM_CLASSES
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.conv1(x)) # BS, C, H, W output size = (W-F+2P)/S + 1 = (100-3+2)/1 + 1 = 100
        x = self.pool(x) # BS, C, H, W output size = (W-F+2P)/S + 1 = (100-2+2)/2 + 1 = 50 input channel = 6, output channel = 6
        x = self.relu(self.conv2(x)) 
        x = self.pool(x)
        x = self.relu(self.conv3(x))# BS, C, H, W output size = (W-F+2P)/S + 1 = (50-3+2)/1 + 1 = 50 input channel = 16, output channel = 16
        x = self.pool(x) # BS, C, H, W output size = (W-F+2P)/S + 1 = (50-2+2)/2 + 1 = 25 input channel = 32, output channel = 32
        x = self.flatten(x) 
        x = self.fc1(x)
        x = self.relu(x) 
        x = self.relu(self.fc2(x)) 
        x = self.relu(self.fc3(x)) 
        x = self.fc4(x) 
        x = self.softmax(x) 

        return x

model = ImageMulticlassClassificationNet()    
#%% model summary
input = torch.rand(1, 1, 100, 100) # BS, C, H, W  
model(input).shape
# test_summary = model(input)
# summary(test_summary)

# %% loss function and optimizer
# set up loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# %% training
# NUM_EPOCHS = 25
# for epoch in range(NUM_EPOCHS):
#     for i, data in enumerate(trainloader, 0):
#         inputs, labels = data
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = loss_fn(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#     print(f'Epoch {epoch}/{NUM_EPOCHS}, Loss: {loss.item():.4f}')
#%%
model = ImageMulticlassClassificationNet()
model.load_state_dict(torch.load("model_dogs1.pt", map_location=torch.device('cpu')))
model.eval()

# %% test
y_test = []
y_test_hat = []
for i, data in enumerate(testloader, 0):
    inputs, y_test_temp = data
    with torch.no_grad():
        y_test_hat_temp = model(inputs).round()
    
    y_test.extend(y_test_temp.numpy())
    y_test_hat.extend(y_test_hat_temp.numpy())

# %%
acc = accuracy_score(y_test, np.argmax(y_test_hat, axis=1))
print(f'Accuracy: {acc*100:.2f} %')
# %% confusion matrix
confusion_matrix(y_test, np.argmax(y_test_hat, axis=1))

#%%
# Load the image and apply the transforms
image = Image.open('multi_2.jpg')
image_tensor = transform(image)

# Pass the image through the model to get the predictions
with torch.no_grad():
    output = model(image_tensor.unsqueeze(0))

# Convert the output to probabilities and extract the class indices
probabilities = torch.softmax(output, dim=1)[0]
class_indices = torch.argsort(probabilities, descending=True)[:3]

# Define the classes

# Count the number of objects in each class
counts = [0, 0, 0]
for i in range(len(class_indices)):
    index = class_indices[i].item()
    counts[i] = int(probabilities[index].item() * image_tensor.numel())

# Print the counts for each class
print('Counts:')
for i in range(NUM_CLASSES):
    print(f'{CLASSES[i]}: {counts[i]}')

# Print the total number of objects
total_count = sum(counts)
print(f'Total count: {total_count}')

# %%
image = Image.open('multi_1.jpg')
image_tensor = transform(image)
# Step 3: Pass the image tensor through the model to obtain a prediction tensor
with torch.no_grad():
    toy_predictions = model(image_tensor.unsqueeze(0))
# Step 4: Convert the prediction tensor to a numpy array and obtain the count of each toy class using numpy functions
print(toy_predictions)
toy_counts = np.bincount(np.argmax(toy_predictions.detach().numpy(), axis=1))

# Step 5: Sum the counts of the three toy groups to obtain the total number of toys in the image
total_toy_count = np.sum(toy_counts)
# %%
total_toy_count
# %%
