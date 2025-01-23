#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# In[2]:


import os
import random
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Initial previous code
data_dir = r"..\Datasets\kvasir-dataset-v2"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_subset_dataset(data_dir, transform, num_samples=20):
    class_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    selected_images = []

    for class_dir in class_dirs:
        images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.endswith(('.jpg', '.png', '.jpeg'))]
        sampled_images = random.sample(images, min(num_samples, len(images)))
        for img_path in sampled_images:
            label = os.path.basename(class_dir)
            selected_images.append((img_path, label))
    
    return selected_images

class CustomDataset(datasets.VisionDataset):
    def __init__(self, selected_images, class_to_idx, transform=None):
        super().__init__(root=None, transform=transform)
        self.selected_images = selected_images
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.selected_images)

    def __getitem__(self, idx):
        img_path, label = self.selected_images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label_idx = self.class_to_idx[label]
        return image, label_idx

selected_images = get_subset_dataset(data_dir, transform)
class_names = [os.path.basename(d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

subset_dataset = CustomDataset(selected_images, class_to_idx, transform=transform)
train_loader = DataLoader(subset_dataset, batch_size=32, shuffle=True)

print(f"Class names: {class_names}")

# Function to create the test subset
def create_test_subset(data_dir, class_to_idx, num_classes=8, num_samples_per_class=2):
    class_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    selected_images = []

    for class_dir in class_dirs:
        images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.endswith(('.jpg', '.png', '.jpeg'))]
        sampled_images = random.sample(images, min(num_samples_per_class, len(images)))
        for img_path in sampled_images:
            label = os.path.basename(class_dir)
            selected_images.append((img_path, label))

    # Ensure we have 16 images in total, combining all classes
    excess_images = len(selected_images) - (num_classes * num_samples_per_class)
    if excess_images > 0:
        selected_images = selected_images[:-excess_images]
    
    return selected_images

# Creating the test subset
test_selected_images = create_test_subset(data_dir, class_to_idx)
test_subset_dataset = CustomDataset(test_selected_images, class_to_idx, transform=transform)
test_loader = DataLoader(test_subset_dataset, batch_size=32, shuffle=True)

print(f"Test subset created with {len(test_selected_images)} images")


# In[3]:


# 2. Feature Extractor (Base Network)
def build_base_network():
    base_model = models.resnet18(pretrained=True)
    base_model.fc = nn.Identity()  # Remove the fully connected layer
    return base_model


# In[4]:


# 3. Compute Class Prototypes
def compute_class_prototypes(train_loader, base_network, class_names):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_network.eval()
    base_network.to(device)

    embeddings = {class_name: [] for class_name in class_names}

    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc="Computing Class Prototypes"):
            images, labels = images.to(device), labels.to(device)
            features = base_network(images)
            for i, label in enumerate(labels):
                class_name = class_names[label.item()]
                embeddings[class_name].append(features[i].cpu())

    class_prototypes = {class_name: torch.mean(torch.stack(embeddings[class_name]), dim=0)
                        for class_name in class_names}
    return class_prototypes


# In[5]:


# 4. Siamese Neural Network Definition
class SiameseNetwork(nn.Module):
    def __init__(self, base_network):
        super(SiameseNetwork, self).__init__()
        self.base_network = base_network
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        img1, prototype = inputs  # img1 is a batch of images, prototype is 1D or expanded
        feature1 = self.base_network(img1)  # Extract features from batch of images
        distance = torch.abs(feature1 - prototype)  # Calculate absolute difference
        similarity = self.fc(distance)  # Calculate similarity
        return similarity


# In[6]:


def train_snn_classification(model, train_loader, class_prototypes, class_names, epochs=10, save_path="snn_model.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            batch_size = images.size(0)
            scores = torch.zeros(batch_size, len(class_names)).to(device)

            # Compute similarity scores with all class prototypes
            for i, class_name in enumerate(class_names):
                prototype = class_prototypes[class_name].to(device)
                similarities = model([images, prototype.expand(batch_size, -1)])
                scores[:, i] = similarities.squeeze()

            # Compute loss and backpropagate
            loss = criterion(scores, labels)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader)}")

    # Ensure save path directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return model


# In[7]:


# 6. Inference Function
def classify_image_snn(image_path, model, class_prototypes, class_names):
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert('RGB')
    input_img = transform(img).unsqueeze(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_img = input_img.to(device)
    model = model.to(device)

    similarity_scores = {}
    with torch.no_grad():
        for class_name, prototype in class_prototypes.items():
            prototype = prototype.unsqueeze(0).to(device)
            similarity = model([input_img, prototype])
            similarity_scores[class_name] = similarity.item()

    predicted_class = max(similarity_scores, key=similarity_scores.get)
    print(f"Similarity scores: {similarity_scores}")
    print(f"Predicted Class: {predicted_class}")

    plt.imshow(np.array(img))
    plt.title(f'Predicted Class: {predicted_class}')
    plt.axis('off')
    plt.show()

    return predicted_class


# In[8]:


# 7. Evaluation function
# Evaluate the model on the test subset and calculate accuracy
def evaluate_model(test_loader, model, class_prototypes, class_names):
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            for img, target in zip(images, labels):
                # Save the image temporarily to disk (just for the inference function)
                temp_image_path = "temp_img.jpg"
                transforms.functional.to_pil_image(img).save(temp_image_path)

                # Use the existing classify_image_snn function
                predicted_class = classify_image_snn(temp_image_path, model, class_prototypes, class_names)
                predicted_class_idx = class_names.index(predicted_class)

                all_preds.append(predicted_class_idx)
                all_targets.append(target.item())

    accuracy = accuracy_score(all_targets, all_preds) * 100
    print(f"Model accuracy on the test subset: {accuracy:.2f}%")


# In[9]:


if __name__ == "__main__":
    # Build the base network
    base_network = build_base_network()

    # Compute class prototypes
    print("Computing class prototypes...")
    class_prototypes = compute_class_prototypes(train_loader, base_network, class_names)

    # Save the prototypes
    torch.save(class_prototypes, "class_prototypes.pth")
    print("Class prototypes saved.")

    # Initialize and train the Siamese Network
    siamese_model = SiameseNetwork(base_network)
    print("Training the Siamese Network...")
    trained_model = train_snn_classification(
        siamese_model, 
        train_loader, 
        class_prototypes, 
        class_names, 
        epochs=10, 
        save_path="./models/snn_model.pth"
    )


# In[10]:


# Evaluating on the test subset.
evaluate_model(test_loader, trained_model, class_prototypes, class_names)

