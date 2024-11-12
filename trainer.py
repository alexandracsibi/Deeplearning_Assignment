import torch
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score

class Trainer:
    def __init__(self, model, optimizer, device='cpu', save_path='best_model.pth'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.save_path = save_path
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def train_epoch(self, data):
        self.model.train()
        self.optimizer.zero_grad()

        # Encode node embeddings
        data = data.to(self.device)
        z = self.model.encode(data.x, data.edge_index)

        # Decode edge predictions
        pos_out = self.model.decode(z, data.pos_edge_label_index).view(-1)
        neg_out = self.model.decode(z, data.neg_edge_label_index).view(-1)

        # Compute loss
        pos_loss = self.criterion(pos_out, data.pos_edge_label.float())
        neg_loss = self.criterion(neg_out, data.neg_edge_label.float())
        loss = pos_loss + neg_loss
        loss.backward()
        self.optimizer.step()
        return loss.item(), pos_loss.item(), neg_loss.item()

    @torch.no_grad()
    def evaluate(self, data):
        self.model.eval()

        data = data.to(self.device)
        z = self.model.encode(data.x, data.edge_index)

        pos_out = self.model.decode(z, data.pos_edge_label_index).view(-1)
        neg_out = self.model.decode(z, data.neg_edge_label_index).view(-1)
        scores = torch.cat([pos_out, neg_out]).sigmoid()
        labels = torch.cat([data.pos_edge_label, data.neg_edge_label])

        y_true = labels.cpu().numpy()
        y_pred_probs = scores.cpu().numpy()

        auc = roc_auc_score(y_true, y_pred_probs)

        # Dynamically evaluate performance across thresholds
        best_f1, best_threshold, best_cm = 0, 0, None

        for threshold in [i / 100 for i in range(1, 100)]:
            y_pred = (y_pred_probs >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)
            if f1  > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_cm = cm

        return {
            'f1': best_f1,
            'auc': auc,
            'threshold': best_threshold,
            'cm': best_cm
        }

    def fit(self, train_data, val_data, num_epochs=100, early_stopping_patience=10):
        best_val_f1 = 0
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            train_loss, pos_loss, neg_loss = self.train_epoch(train_data)
            val_metric = self.evaluate(val_data)

            # Save the best model
            if val_metric['f1'] > best_val_f1:
                best_val_f1 = val_metric['f1']
                patience_counter = 0
                torch.save({'model_state_dict': self.model.state_dict(),
                'best_threshold': val_metric['threshold']}, self.save_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print("Early stopping triggered.")
                    break
            self.scheduler.step()
            if epoch % 5 == 0:
                print(f"{'Epoch':<6} {'Train Loss':<12} {'Pos Loss':<10} {'Neg Loss':<10} {'Val F1':<8} {'Val AUC':<9} {'Threshold':<10}")
                print(f"  {epoch:<6} {train_loss:<12.4f} {pos_loss:<10.4f} {neg_loss:<10.4f} {val_metric['f1']:<8.4f} {val_metric['auc']:<9.4f} {val_metric['threshold']:<10.2f}")
                print(f"{'=' * 70}")
                print(f"Confusion Matrix:\n{val_metric['cm']}")
                print(f"{'=' * 70}")

    def test(self, test_data):
        checkpoint = torch.load(self.save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        best_threshold = checkpoint['best_threshold']
        test_metric = self.evaluate(test_data)
        print(f"Test F1: {test_metric['f1']:.4f}")
        print(f"Test AUC: {test_metric['auc']:.4f}")
        print(f"Best Threshold: {best_threshold:.2f}")
        print(f"Confusion Matrix:\n{test_metric['cm']}")