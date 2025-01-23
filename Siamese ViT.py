#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms, models 
from PIL import Image
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# In[2]:


# Initial setup
data_dir = r"..\Datasets\kvasir-dataset-v2"
all_class_names = [os.path.basename(d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))] 
class_names = [class_name for class_name in all_class_names if class_name != "polyps"]
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
        
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the full dataset
full_dataset = datasets.ImageFolder(data_dir, transform=transform)
classes = full_dataset.classes
class_to_idx = full_dataset.class_to_idx

# Split the dataset into training and testing sets
test_ratio = 0.2
test_size = int(len(full_dataset) * test_ratio)
train_size = len(full_dataset) - test_size

all_indices = list(range(len(full_dataset)))
random.shuffle(all_indices)

train_indices, test_indices = all_indices[:train_size], all_indices[train_size:]

train_subset = Subset(full_dataset, train_indices)
test_subset = Subset(full_dataset, test_indices)

# Ensure training dataset has no polyps
polyps_class = "polyps"
train_indices_without_polyps = [i for i in train_indices if full_dataset.imgs[i][1] != class_to_idx[polyps_class]]

train_subset_without_polyps = Subset(full_dataset, train_indices_without_polyps)

# Create a small polyps dataset with 20 images
def get_polyps_dataset(data_dir, transform, num_samples=20, target_class="polyps"):
    polyps_dir = os.path.join(data_dir, target_class)
    polyps_images = [os.path.join(polyps_dir, img) for img in os.listdir(polyps_dir) if img.endswith(('.jpg', '.png', '.jpeg'))]
    sampled_images = random.sample(polyps_images, min(num_samples, len(polyps_images)))

    selected_images = [(img_path, target_class) for img_path in sampled_images]
    return selected_images

polyps_images = get_polyps_dataset(data_dir, transform)
polyps_dataset = CustomDataset(polyps_images, class_to_idx, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_subset_without_polyps, batch_size=32, shuffle=True)
polyps_loader = DataLoader(polyps_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=32, shuffle=True)

print(f"Number of samples in the training dataset without polyps: {len(train_subset_without_polyps)}")
print(f"Number of samples in the polyps dataset: {len(polyps_dataset)}")
print(f"Number of samples in the test dataset: {len(test_subset)}")


# In[3]:


# 2. Feature Extractor (Base Network)
def build_base_network():
    base_model = timm.create_model('vit_small_patch16_224', pretrained=True)  # Load pre-trained ViT model
    base_model.head = nn.Identity()  # Remove the classification head
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

    # Ensure no empty tensors are passed to torch.stack
    class_prototypes = {class_name: torch.mean(torch.stack(embeddings[class_name]), dim=0)
                        for class_name in class_names if len(embeddings[class_name]) > 0}
    
    # Handle missing classes
    for class_name in class_names:
        if class_name not in class_prototypes:
            print(f"Warning: No embeddings found for class {class_name}. Skipping prototype computation for this class.")

    return class_prototypes

# Function to compute prototype for polyps during few-shot learning
def compute_polyps_prototype(polyps_loader, base_network):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_network.eval()
    base_network.to(device)

    polyps_embeddings = []

    with torch.no_grad():
        for images, _ in tqdm(polyps_loader, desc="Computing Polyps Prototype"):
            images = images.to(device)
            features = base_network(images)
            polyps_embeddings.append(features.cpu())

    polyps_prototype = torch.mean(torch.stack(polyps_embeddings), dim=0)
    return polyps_prototype


# In[5]:


# 4. Siamese Neural Network Definition
class SiameseNetwork(nn.Module):
    def __init__(self, base_network):
        super(SiameseNetwork, self).__init__()
        self.base_network = base_network
        self.fc = nn.Sequential(
            nn.Linear(base_network.embed_dim, 128),  # Match input dimension to ViT output
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
    # Compute class prototypes on training dataset without polyps
    print("Computing class prototypes...")
    base_network = build_base_network()
    class_prototypes = compute_class_prototypes(train_loader, base_network, class_names)
    torch.save(class_prototypes, "class_prototypes.pth")
    print("Class prototypes saved.")

    # Initialize and train the Siamese Network on non-polyps training dataset
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

    # Evaluate the model on the test dataset
    evaluate_model(test_loader, trained_model, class_prototypes, all_class_names)

    # Fine-tune the trained model on the small polyps dataset
    print("Fine-tuning the model on the polyps dataset...")
    few_shot_model = train_snn_classification(
        trained_model, 
        polyps_loader, 
        class_prototypes, 
        all_class_names, 
        epochs=5,  # Fine-tune for a few epochs
        save_path="./models/snn_finetuned_polyps.pth"
    )

    # Evaluate the fine-tuned model on the test dataset
    evaluate_model(test_loader, few_shot_model, class_prototypes, all_class_names)

