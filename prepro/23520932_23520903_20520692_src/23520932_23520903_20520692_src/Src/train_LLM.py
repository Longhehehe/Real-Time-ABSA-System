import sys 
import os
sys.path.insert(0, '.')
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from Src.preprocessing import preprocess_text_for_llm
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = [str(text) for text in texts]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SentimentClassifier(nn.Module):
    """Custom PyTorch model with transformer backbone and FC head"""
    def __init__(self, model_name, num_classes, dropout_rate=0.1):
        super(SentimentClassifier, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        logits = model(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = F.cross_entropy(logits, labels)
            
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_llm_with_preprocessed(data, preprocessed_texts, model_name='vinai/phobert-base', max_length=256, 
                               epochs=3, batch_size=32, learning_rate=2e-5, output_dir="./llm_models"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unique_labels = sorted(data['label'].unique())
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    data['label_int'] = data['label'].map(label_to_int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        preprocessed_texts, data['label_int'].tolist(), 
        test_size=0.2, random_state=42, stratify=data['label_int']
    )
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    train_dataset = SentimentDataset(X_train, y_train, tokenizer, max_length)
    test_dataset = SentimentDataset(X_test, y_test, tokenizer, max_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Initialize model
    model = SentimentClassifier(model_name, len(unique_labels))
    model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=total_steps
    )
    
    # Training loop
    best_f1 = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        
        # Evaluate
        eval_results = evaluate_model(model, test_loader, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Eval Loss: {eval_results['loss']:.4f}")
        print(f"Eval Accuracy: {eval_results['accuracy']:.4f}")
        print(f"Eval F1: {eval_results['f1']:.4f}")
        
        # Save best model
        if eval_results['f1'] > best_f1:
            best_f1 = eval_results['f1']
            # Save model
            os.makedirs(output_dir, exist_ok=True)
            model_path = f"{output_dir}/final_model"
            os.makedirs(model_path, exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'tokenizer': tokenizer,
                'model_name': model_name,
                'num_classes': len(unique_labels),
                'label_to_int': label_to_int,
                'max_length': max_length
            }, f"{model_path}/model.pt")
            
            tokenizer.save_pretrained(model_path)
    
    print(f"Training completed! Best F1: {best_f1:.4f}")
    
    # Return results
    test_results = {
        'test_accuracy': eval_results['accuracy'],
        'test_f1': eval_results['f1'],
        'test_precision': eval_results['precision'],
        'test_recall': eval_results['recall']
    }
    
    return model, tokenizer, test_results

def load_model(model_path):
    """Load trained model"""
    checkpoint = torch.load(f"{model_path}/model.pt", map_location='cpu')
    
    model = SentimentClassifier(
        checkpoint['model_name'], 
        checkpoint['num_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer, checkpoint

class LLMSentimentClassifier:
    def __init__(self, model_name: str, vncorenlp_path: str = "./VnCoreNLP/VnCoreNLP-1.1.1.jar"):
        self.model_name = model_name
        self.vncorenlp_path = vncorenlp_path
        self.model = None
        self.tokenizer = None
        self.checkpoint = None
        self.is_fitted = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_model_with_preprocessed(self, data, preprocessed_texts, epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5, max_length: int = 256):
        start_time = time.time()
        print(f"Using {len(preprocessed_texts)} preprocessed texts for LLM training")
        
        # Train model using custom train_llm_with_preprocessed function
        self.model, self.tokenizer, test_results = train_llm_with_preprocessed(
            data=data,
            preprocessed_texts=preprocessed_texts,
            model_name=self.model_name,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=max_length
        )
        
        self.model.to(self.device)
        training_time = time.time() - start_time
        
        # Format results to match traditional ML interface
        results = {
            'accuracy': test_results['test_accuracy'],
            'precision': test_results['test_precision'],
            'recall': test_results['test_recall'],
            'f1': test_results['test_f1'],
            'training_time': training_time
        }
        
        self.is_fitted = True
        
        return results
    
    def predict(self, texts):
        # Preprocess texts
        if isinstance(texts, str):
            texts = [texts]
        
        preprocessed_texts = preprocess_text_for_llm(
            texts, 
            use_word_segmentation=True, 
            use_parallel=False
        )
        
        # Make predictions
        predictions = []
        self.model.eval()
        
        with torch.no_grad():
            for text in preprocessed_texts:
                inputs = self.tokenizer(
                    text, 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True, 
                    max_length=256
                )
                
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                prediction = torch.argmax(logits, dim=1)
                predictions.append(prediction.item())
        
        return predictions

