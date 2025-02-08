# Cross Sentiment - Sentiment Analysis Model
# ==========================================
# 
# Written by Shreeya Bekkam, Jake Huseman, and Aamandra Sandeep Thakur.
# 
# Originally created for CS5134/6034 Natural Language Processing at the
# University of Cincinnati. Updated in February 2025 to improve memory
# usage and readability.


import os
import pandas as pd
import re
import torch
import numpy as np
from io import StringIO
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertModel, BertTokenizer, BertModel
from torch_geometric.nn import GCNConv
from penman import decode as decode_amr
from penman.models.noop import Model
from torch import nn
from torch.optim import Adam
import amrlib
from penman import Graph
from colorama import Fore, Style
from torch_geometric.nn import GCNConv


class SentimentClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SentimentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128) # Hidden layer (|input| x 128).
        self.fc2 = nn.Linear(128, 64)        # Hidden layer (128 x 64).
        self.fc3 = nn.Linear(64, 2)          # Output layer (binary classification).
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CrossAttentionModule(nn.Module):
    def __init__(self, text_dim, graph_dim, output_dim):
        super(CrossAttentionModule, self).__init__()

        # Project text embeddings to output dimension.
        self.text_proj = nn.Linear(text_dim, output_dim)

        # Project graph embeddings to output dimension.
        self.graph_proj = nn.Linear(graph_dim, output_dim)

        # Attention projection layer.
        self.attn_proj = nn.Linear(output_dim, 1)

    def forward(self, text_embeddings, graph_embeddings):
        # Project text embeddings.
        text_proj = self.text_proj(text_embeddings)

        # Project graph embeddings.
        graph_proj = self.graph_proj(graph_embeddings)
        
        # Combine the projected embeddings.
        combined = text_proj + graph_proj

        # Compute attention weights.
        attn_weights = torch.softmax(self.attn_proj(combined), dim=0)

        # Apply attention weights to the combined embeddings.
        attended_features = attn_weights * combined
        return attended_features


# (Ensure AMR model is properly loaded.)
model_dir = 'model_parse_xfm_bart_large-v0_1_0'
stog = amrlib.load_stog_model(model_dir=model_dir)
print("Loaded AMR parsing model.")


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


# Load pre-trained BERT for node embeddings.
node_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
node_model = BertModel.from_pretrained('bert-base-uncased')


def get_node_embeddings(node_labels):
    print(f"Generating embeddings for nodes: {list(node_labels.values())[:3]}...")

    # Extract labels as strings.
    label_list = list(node_labels.values())
    inputs = node_tokenizer(label_list, return_tensors='pt', padding=True, truncation=True)
    outputs = node_model(**inputs)

    # Return mean-pooled embeddings.
    return outputs.last_hidden_state.mean(dim=1)


def sanitize_xml_content(file_path):
    print(f"Sanitizing XML content from file: {file_path}")
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()
    content = re.sub(r'[^\x09\x0A\x0D\x20-\x7F]', '', content)
    content = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;|#[0-9]+;)', '&amp;', content)
    if not content.strip().startswith('<reviews>'):
        content = f"<reviews>{content}</reviews>"
    print("Sanitization complete.")
    return content


def process_xml_file(file_path, sentiment_label):
    try:
        print(f"Processing XML file: {file_path} with sentiment label: {sentiment_label}")
        xml_content = sanitize_xml_content(file_path)
        xml_stream = StringIO(xml_content)
        df = pd.read_xml(xml_stream, xpath='.//review', parser='lxml')
        df['sentiment'] = sentiment_label
        for column in df.columns:
            df[column] = df[column].astype(str).str.strip().str.replace(r'\n', '', regex=True)
        print(f"Processed file: {file_path}. DataFrame shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return pd.DataFrame()


def process_category(folder_path, category):
    print(f"Processing category: {category}")
    positive_file = os.path.join(folder_path, f"positive_{category}.review")
    negative_file = os.path.join(folder_path, f"negative_{category}.review")
    all_reviews = []
    if os.path.exists(positive_file):
        print(f"Found positive file for {category}: {positive_file}")
        positive_reviews = process_xml_file(positive_file, sentiment_label=1)
        all_reviews.append(positive_reviews)
    if os.path.exists(negative_file):
        print(f"Found negative file for {category}: {negative_file}")
        negative_reviews = process_xml_file(negative_file, sentiment_label=0)
        all_reviews.append(negative_reviews)
    combined_df = pd.concat(all_reviews, ignore_index=True) if all_reviews else pd.DataFrame()
    print(f"Completed processing category: {category}. Combined DataFrame shape: {combined_df.shape}")
    return combined_df


# Preprocess datasets.
folder_path = 'Dataset'
books_df = process_category(folder_path, 'books')
dvd_df = process_category(folder_path, 'dvd')
electronics_df = process_category(folder_path, 'electronics')
kitchen_df = process_category(folder_path, 'kitchen')


def shorten_text(text):
    for i in range(len(text)):
        if text[i] == ".":
            return text[:i] + "."
    return text[:256]


# Limit the number of samples in each DataFrame.
books_df = books_df.sample(n=80, random_state=42)              # Take N random samples from books
dvd_df = dvd_df.sample(n=2, random_state=42)                   # Take N random samples from DVD
electronics_df = electronics_df.sample(n=50, random_state=42)  # Take N random samples from electronics
kitchen_df = kitchen_df.sample(n=2, random_state=42)           # Take N random samples from kitchen

print(f"Books DataFrame reduced to {len(books_df)} samples")
print(f"DVD DataFrame reduced to {len(dvd_df)} samples")
print(f"Electronics DataFrame reduced to {len(electronics_df)} samples")
print(f"Kitchen & Housewares DataFrame reduced to {len(kitchen_df)} samples")

print("Processed all categories.")
print("Books DataFrame sample:\n", books_df.head())
print("Electronics DataFrame sample:\n", electronics_df.head())


def parse_to_amr(text):
    try:
        print(f"Parsing text to AMR: {text[:50]}...")  # Print first 50 chars for brevity
        amr_graphs = stog.parse_sents([text])
        amr = decode_amr(amr_graphs[0], model=Model()) if amr_graphs[0] else None
        print("AMR parsing successful.")
        return amr
    except Exception as e:
        print(f"Error parsing sentence: {text}, Error: {e}")
        return None


def extract_amr_features(amr_graph: Graph):
    # Extract all nodes from source and target in triples.
    nodes = {triple[0] for triple in amr_graph.triples} | {triple[2] for triple in amr_graph.triples}
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}  # Map node to index

    # Filter node labels.
    node_labels = {
        node_to_idx[triple[0]]: triple[2]
        for triple in amr_graph.triples if triple[0] in node_to_idx
    }

    # Extract edges: filter only valid edges.
    edges = [
        (node_to_idx.get(triple[0], -1), node_to_idx.get(triple[2], -1))
        for triple in amr_graph.triples if triple[1].startswith(':')
    ]

    # Remove invalid edges where -1 is used (i.e., missing nodes).
    edges = [(src, tgt) for src, tgt in edges if src != -1 and tgt != -1]

    # Check if the edge indices are within bounds of the node features.
    num_nodes = len(node_labels)
    edges = [(src, tgt) for src, tgt in edges if src < num_nodes and tgt < num_nodes]

    # Convert edges to edge_index for PyTorch Geometric.
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    
    return node_labels, edge_index


# Function to encode texts using DistilBERT.
def encode_texts(texts, tokenizer, model, batch_size=16, device='cpu'):
    print(f"Encoding {len(texts)} texts with DistilBERT...")
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        print(f"Encoding batch {i // batch_size + 1}: {len(batch_texts)} texts...")
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1)) # Mean pooling.
    print("Encoding complete.")
    return torch.cat(embeddings, dim=0)


def train_on_dataset(train_df, test_df, device='cpu'):
    print("Starting training...")
    train_df['sentiment'] = pd.to_numeric(train_df['sentiment'], errors='coerce')
    test_df['sentiment'] = pd.to_numeric(test_df['sentiment'], errors='coerce')
    train_texts = train_df['review_text'].tolist()
    test_texts = test_df['review_text'].tolist()
    print(f"Training data size: {len(train_texts)}")
    print(f"Testing data size: {len(test_texts)}")
    
    # DistilBERT text embeddings.
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
    train_text_embeddings = encode_texts(train_texts, tokenizer, bert_model, device=device)
    print(f"Train embeddings shape: {train_text_embeddings.shape}")
    test_text_embeddings = encode_texts(test_texts, tokenizer, bert_model, device=device)
    print(f"Test embeddings shape: {test_text_embeddings.shape}")

    # Prepare AMR embeddings with GNN.
    train_amr_embeddings = []
    i = 0
    for text in train_texts:
        i += 1
        print(Fore.MAGENTA + "Preparing training text " , i, Style.RESET_ALL)
        print(f"Processing text for GNN: {text[:50]}...")
        amr = parse_to_amr(text)
        if amr:
            node_labels, edge_index = extract_amr_features(amr)
            if len(node_labels) > 0:  # Check if valid nodes exist
                node_features = get_node_embeddings(node_labels).to(device)
                gcn = GCN(input_dim=768, hidden_dim=64, output_dim=32).to(device)
                amr_embedding = gcn(node_features, edge_index.to(device)).mean(dim=0)
                train_amr_embeddings.append(amr_embedding)
                continue
        print(Fore.MAGENTA + "  (failed to get AMR embeddings.)", Style.RESET_ALL)
        train_amr_embeddings.append(torch.zeros(32, device=device))  # Placeholder for AMR embedding
    train_amr_embeddings = torch.stack(train_amr_embeddings)
    print(f"Train AMR embeddings shape: {train_amr_embeddings.shape}")

    # Cross-Attention and Classification.
    cross_attention = CrossAttentionModule(text_dim=768, graph_dim=32, output_dim=256).to(device)
    train_combined = cross_attention(train_text_embeddings, train_amr_embeddings)
    classifier = SentimentClassifier(input_dim=256).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(classifier.parameters(), lr=0.01)

    print("Starting training loop...")
    for epoch in range(40):
        classifier.train()

        # Clear gradients.
        optimizer.zero_grad()
        train_combined = cross_attention(train_text_embeddings.detach(), train_amr_embeddings.detach())

        # Forward pass.
        outputs = classifier(train_combined)
        loss = criterion(outputs, torch.tensor(train_df['sentiment'].values.astype(int), dtype=torch.long).to(device))
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

        loss.backward()   # Backpropagation.
        optimizer.step()  # Update weights.

    classifier.eval()
    print("Starting testing...")
    with torch.no_grad():
        test_amr_embeddings = []
        i = 0
        for text in test_texts:
            i += 1
            print(Fore.MAGENTA + "Preparing testing text " , i, Style.RESET_ALL)
            print(f"Processing test text for GNN: {text[:50]}...")
            amr = parse_to_amr(text)
            if amr:
                node_labels, edge_index = extract_amr_features(amr)
                if len(node_labels) > 0:
                    node_features = get_node_embeddings(node_labels).to(device)
                    gcn = GCN(input_dim=768, hidden_dim=64, output_dim=32).to(device)
                    amr_embedding = gcn(node_features, edge_index.to(device)).mean(dim=0)
                    test_amr_embeddings.append(amr_embedding)
                    continue
            print(f"AMR parsing failed for test text: {text[:50]}. Using placeholder.")
            test_amr_embeddings.append(torch.zeros(32, device=device))
        
        test_amr_embeddings = torch.stack(test_amr_embeddings)
        print(f"Test AMR embeddings shape: {test_amr_embeddings.shape}")

        # Compute combined features for test data.
        test_combined = cross_attention(test_text_embeddings, test_amr_embeddings)

        # Forward pass through the classifier.
        predictions = torch.argmax(classifier(test_combined), dim=1)
        accuracy = (predictions == torch.tensor(test_df['sentiment'].values.astype(int), dtype=torch.long).to(device)).sum().item() / len(test_df) * 100

        num_tp = 0
        num_tn = 0
        num_fp = 0
        num_fn = 0
        for i in range(len(test_df["sentiment"])):
            gold = (test_df["sentiment"].values.astype(int))[i]
            prediction = predictions[i]

            num_tp += 1 if gold == 1 and prediction == 1 else 0
            num_tn += 1 if gold == 0 and prediction == 0 else 0
            num_fp += 1 if gold == 0 and prediction == 1 else 0
            num_fn += 1 if gold == 1 and prediction == 0 else 0

        print(f"Test Accuracy: {accuracy:.2f}%")

        prec = "DIV ZERO" if num_tp + num_fp == 0 else float(num_tp)/(num_tp + num_fp)
        recall = "DIV ZERO" if num_tp + num_fn == 0 else float(num_tp)/(num_tp + num_fn)
        acc = float(num_tp + num_tn)/(num_tp + num_fp + num_tn + num_fn)
        f1 = "DIV ZERO" if prec == "DIV ZERO" or recall == "DIV ZERO" else (2 * prec * recall) / (prec + recall)

        print(Fore.GREEN, f"( tp: {num_tp}, tn: {num_tn}, fp: {num_fp}, fn: {num_fn})", Style.RESET_ALL)
        print(Fore.GREEN, f"precision : {prec}", Style.RESET_ALL)
        print(Fore.GREEN, f"recall    : {recall}", Style.RESET_ALL)
        print(Fore.GREEN, f"accuracy  : {acc}", Style.RESET_ALL)
        print(Fore.GREEN, f"f1 score  : {f1}", Style.RESET_ALL)


# For example: Train on Books. Test on Electronics.
train_on_dataset(books_df, electronics_df, device='cuda' if torch.cuda.is_available() else 'cpu')
