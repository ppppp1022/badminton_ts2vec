import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import List, Dict
import os
from processed_data_loader import ProcessedBadmintonDataset
from preprocess_badminton_data import load_skill_levels_from_annotation
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm 


# ============================================================================
# 1. TS2Vec 모델 구현
# ============================================================================

class DilatedConvEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth=10):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        
        for i in range(depth):
            dilation = 2 ** i
            # padding을 dilation만큼 주어 길이를 유지 (Chomp1d 방식이 더 정확하지만 여기선 padding으로 대체)
            self.conv_layers.append(
                nn.Conv1d(
                    input_dim if i == 0 else hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    padding=dilation, 
                    dilation=dilation
                )
            )
            
    def forward(self, x):
        # x: (batch, length, input_dim)
        x = x.transpose(1, 2)  # (batch, input_dim, length)
        
        for i, conv in enumerate(self.conv_layers):
            residual = x
            x = conv(x)
            x = F.relu(x)
            
            # [수정] 차원이 같을 때만 Residual 연결
            if x.shape[1] == residual.shape[1]:
                x = x + residual
                
        return x.transpose(1, 2)  # (batch, length, hidden_dim)
    
class TS2Vec(nn.Module):
    """TS2Vec 메인 모델"""
    def __init__(self, input_dim, hidden_dim=64, depth=10, output_dim=320):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.encoder = DilatedConvEncoder(input_dim, hidden_dim, depth)
        
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x, mask=None):
        # x: (batch, length, input_dim)
        encoded = self.encoder(x)  # (batch, length, hidden_dim)
        
        if mask is not None:
            encoded = encoded * mask.unsqueeze(-1)
        
        projected = self.projector(encoded)
        
        return encoded, projected
    
    def encode(self, x, mask=None):
        """추론용: representation 추출"""
        with torch.no_grad():
            encoded, _ = self.forward(x, mask)
            # Temporal pooling
            if mask is not None:
                encoded = (encoded * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
            else:
                encoded = encoded.mean(1)
        return encoded

class TS2VecLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z1, z2):
        """
        TS2Vec의 핵심: Instance-wise가 아니라 Timestamp-wise Loss를 계산해야 함
        z1, z2: (batch, length, dim)
        """
        batch_size, seq_len, _ = z1.shape
        
        # [핵심 1] 시간축을 따라 계층적(Hierarchical)으로 Loss 계산
        # 논문에서는 Max Pooling을 하며 반복하지만, 간소화를 위해 Original Scale에서만 계산하는 버전입니다.
        # (제대로 하려면 loop 돌며 max_pool1d 후 loss 합산 필요)
        
        loss = self._compute_loss(z1, z2, batch_size, seq_len)
        return loss

    def _compute_loss(self, z1, z2, batch_size, seq_len):
        # (Batch * Length, Dim) 형태로 펼침 -> 모든 타임스탬프를 개별 샘플로 취급
        z1_flat = z1.reshape(batch_size * seq_len, -1)
        z2_flat = z2.reshape(batch_size * seq_len, -1)
        
        targets = torch.arange(batch_size * seq_len, device=z1.device)
        
        # 같은 타임스탬프끼리 Positive
        # z1의 t시점 <-> z2의 t시점
        sim_matrix = torch.matmul(z1_flat, z2_flat.T) / self.temperature
        
        # 대각 성분(자기 자신과의 유사도)이 정답
        loss = F.cross_entropy(sim_matrix, targets)
        return loss


# ============================================================================
# 2. Skill Level 분류 모델
# ============================================================================

class SkillLevelClassifier(nn.Module):
    """TS2Vec 임베딩으로부터 skill level 분류 (Batch Norm 추가 버전)"""
    def __init__(self, embedding_dim, num_classes=7):
        super().__init__()
        
        self.classifier = nn.Sequential(
            # Layer 1
            nn.Linear(embedding_dim, 256),
            nn.BatchNorm1d(256),       # [추가] 학습 안정화
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),       # [추가]
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),        # [추가]
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 4
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),        # [추가]
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Output Layer
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)

# ============================================================================
# 4. 학습 파이프라인
# ============================================================================


class StrokeDataset(Dataset):
    
    def __init__(self, data_list, labels=None):
        self.strokes = data_list
        self.labels = []
        self.has_labels = labels is not None
        
        # 모든 subject의 stroke들을 하나의 리스트로 펼침
        if self.has_labels:
            self.labels = labels
    
    def __len__(self):
        return len(self.strokes)
    
    def __getitem__(self, idx):
        data = torch.tensor(self.strokes[idx], dtype=torch.float32)
        if self.has_labels:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return data, label
        return data


def collate_fn_with_labels(batch):
    
    padded = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    
    # Mask 생성 - True=진짜 데이터, False=패딩
    mask = (padded.abs().sum(dim=-1) > 0)
    
    return padded, mask

class ImprovedTS2VecTrainer:
    """DataLoader를 사용하는 TS2Vec Trainer"""
    def __init__(self, model, lr=0.001, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # TS2VecLoss는 model과 별도로 정의되어 있다고 가정
        self.loss_fn = TS2VecLoss(temperature=0.5)

    def augment(self, x):
        """
        TS2Vec-style augmentation
        1. Jittering (노이즈)
        2. Scaling (크기)
        3. Masking (구간 가리기)
        
        Args:
            x: (batch, seq_len, features)
        
        Returns:
            x_aug: augmented version
        """
        batch_size, seq_len, feature_dim = x.shape
        x_aug = x.clone()
        
        # --- 1. Jittering (노이즈 추가) ---
        noise = torch.randn_like(x_aug) * 0.03
        x_aug = x_aug + noise
        
        # --- 2. Scaling (크기 변형) ---
        scale = torch.randn(batch_size, 1, 1, device=x.device) * 0.1 + 1
        x_aug = x_aug * scale
        
        # --- 3. Masking (구간 가리기) ---
        # 30% 확률로 전체 길이의 1/4 구간을 0으로 만듦
        if np.random.random() < 0.5:
            mask_len = seq_len // 3
            if mask_len > 0:
                start = np.random.randint(0, max(1, seq_len - mask_len))
                x_aug[:, start:start+mask_len, :] = 0
            
        return x_aug
    
    def train_epoch(self, dataloader):
        total_loss = 0
        num_batches = 0
        total_samples = 0
        
        # 시작 시간 기록
        for batch_idx, (batch_data, mask) in enumerate(dataloader):

            batch_data = batch_data.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            
            # Data Augmentation: 두 개의 view 생성
            x_aug1 = self.augment(batch_data)
            x_aug2 = self.augment(batch_data)
            
            # Forward: 두 augmented view의 representation 추출
            _, z1 = self.model(x_aug1, mask)  # (batch, length, output_dim)
            _, z2 = self.model(x_aug2, mask)  # (batch, length, output_dim)
            
            # TS2Vec Contrastive Loss 계산
            loss = self.loss_fn(z1, z2)
            
            # Backward
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            batch_size_actual = batch_data.size(0)

            total_samples += batch_size_actual

            total_loss += loss.item()

            num_batches += 1

            # 통계 수집
            total_loss += loss.item()
           
        return total_loss / num_batches

# ============================================================================
# 5. 메인 실행 파이프라인
# ============================================================================

class SkillLevelTrainer:
    """Skill Level 분류기 학습"""
    def __init__(self, ts2vec_model, classifier, lr=0.001, device='cuda'):
        self.ts2vec_model = ts2vec_model.to(device)
        self.classifier = classifier.to(device)
        self.device = device
        # 분류기 파라미터만 학습
        self.optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # TS2Vec은 Freeze (학습되지 않게 고정)
        for param in self.ts2vec_model.parameters():
            param.requires_grad = False
        self.ts2vec_model.eval()
        
    def extract_embeddings(self, data_list: List[np.ndarray]) -> np.ndarray:
        self.ts2vec_model.eval()
        
        device = next(self.ts2vec_model.parameters()).device
        embeddings = []

        for stroke in data_list:
            
            stroke_tensor = torch.FloatTensor(stroke).unsqueeze(0).to(device)
            
            mask = torch.ones((1, stroke.shape[0]), dtype=torch.bool).to(device)
            
            with torch.no_grad():
                output = self.ts2vec_model(stroke_tensor, mask)
                
                if isinstance(output, tuple):
                    reprs = output[1] 
                else:
                    reprs = output

                reprs = reprs.transpose(1, 2)
                
                pool_out = F.max_pool1d(reprs, kernel_size=reprs.size(-1))
                
                vector = pool_out.squeeze().cpu().numpy()
                
                embeddings.append(vector)
                
        return np.vstack(embeddings) # 또는 np.vstack(embeddings)
    
    def train_epoch(self, embeddings: np.ndarray, labels: np.ndarray, batch_size=32):
        """1 에폭 학습"""
        self.classifier.train()
        total_loss = 0
        num_batches = 0
        
        # Shuffle
        indices = np.random.permutation(len(embeddings))
        embeddings = embeddings[indices]
        labels = labels[indices]
        
        for i in range(0, len(embeddings), batch_size):
            batch_x = torch.FloatTensor(embeddings[i:i+batch_size]).to(self.device)
            # 라벨: 1~7 -> 0~6 변환
            batch_y = torch.LongTensor(labels[i:i+batch_size] - 1).to(self.device)
            
            # Forward
            outputs = self.classifier(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def evaluate(self, embeddings: np.ndarray, labels: np.ndarray, batch_size=64) -> Dict:
        """모델 평가"""
        self.classifier.eval()
        all_predictions = []
        
        # [팁] 평가 데이터가 많을 경우를 대비해 배치 처리
        with torch.no_grad():
            for i in range(0, len(embeddings), batch_size):
                batch_x = torch.FloatTensor(embeddings[i:i+batch_size]).to(self.device)
                outputs = self.classifier(batch_x)
                preds = outputs.argmax(dim=1).cpu().numpy() + 1 # 0-6 -> 1-7
                all_predictions.append(preds)
                
        predictions = np.concatenate(all_predictions)
        labels = np.array(labels, dtype=int)
        
        # Metrics
        accuracy = accuracy_score(labels, predictions)
        
        # RMSE (순서가 있는 클래스이므로 유의미함)
        mse = mean_squared_error(labels, predictions)
        rmse = np.sqrt(mse)
        
        # Report
        unique_labels = sorted(set(labels) | set(predictions))
        target_names = [f'Level {i}' for i in unique_labels]
        
        report = classification_report(
            labels, 
            predictions, 
            labels=unique_labels,
            target_names=target_names,
            zero_division=0,
            output_dict=True # 딕셔너리 형태로 반환하면 저장하기 편함
        )
        cm = confusion_matrix(labels, predictions, labels=unique_labels)
        
        return {
            'accuracy': accuracy,
            'rmse': rmse,
            'predictions': predictions,
            'report': report,
            'confusion_matrix': cm.tolist() # JSON 저장을 위해 리스트 변환
        }


def main():
    # 설정
    data_folder = './Processed_Data'  # HDF5 파일들이 있는 폴더
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 데이터셋 초기화
    dataset = ProcessedBadmintonDataset(data_folder)
    clear_skills, drive_skills, subject_groups = load_skill_levels_from_annotation(annotation_filepath='./configs/skill_levels.json')
    dataset.beginner_subjects = subject_groups['beginner']
    dataset.intermediate_subjects = subject_groups['intermediate']
    dataset.expert_subjects = subject_groups['expert']
    # TODO: 여기에 skill level 리스트 입력
    dataset.clear_skill_level = [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4]  # 예시
    dataset.drive_skill_level = [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4]  # 예시
    
    # TODO: 여기에 부위별 인덱스 입력
    dataset.local_arm_index = [13, 14, 15, 16, 17, 18, 19, 20]  # 예시
    dataset.global_arm_index = [13, 14, 15, 16, 17, 18, 19, 20]  # 예시
    dataset.local_leg_index = [1, 2, 3, 4, 5, 6]  # 예시
    dataset.global_leg_index = [1, 2, 3, 4, 5, 6]  # 예시
    dataset.local_total_index = list(range(21))
    dataset.global_total_index = list(range(21))
    
    # Stroke type 선택
    stroke_type = 'clear'  # 'clear' or 'drive'
    joint_type = 'global'  # 'local' or 'global'
    body_part = 'arm'  # 'arm', 'leg', 'total'
    
    print(f"Processing: {stroke_type} - {joint_type} - {body_part}")
    
    # 데이터 분할
    train_subjects, test_subjects, train_labels, test_labels = dataset.split_data_by_skill(stroke_type)
    
    print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
    print(f"Test subjects ({len(test_subjects)}): {test_subjects}")
    
    # 학습 데이터 로드
    train_data_list = []
    for subject in train_subjects:
        data, skill = dataset.load_subject_data(subject, stroke_type, joint_type, body_part)
        train_data_list.append(data)
    
    # 테스트 데이터 로드
    test_data_list = []
    for subject in test_subjects:
        data, skill = dataset.load_subject_data(subject, stroke_type, joint_type, body_part)
        test_data_list.append(data)
    
    # Input dimension 계산
    sample_data = train_data_list[0][0]
    input_dim = sample_data.shape[1]
    
    print(f"Input dimension: {input_dim}")
    
    # TS2Vec 모델 생성 및 학습
    print("\n Training TS2Vec ")
    ts2vec_model = TS2Vec(input_dim=input_dim, hidden_dim=64, output_dim=320)
    ts2vec_trainer = TS2VecTrainer(ts2vec_model, lr=0.001, device=device)
    
    for epoch in range(1):
        loss = ts2vec_trainer.train_epoch(train_data_list, batch_size=32)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/100, Loss: {loss:.4f}')
    
    # 임베딩 추출
    print("\n Extracting Embeddings")
    skill_trainer = SkillLevelTrainer(ts2vec_model, 
                                     SkillLevelClassifier(embedding_dim=64, num_classes=7),
                                     lr=0.001, device=device)
    
    train_embeddings = skill_trainer.extract_embeddings(train_data_list)
    test_embeddings = skill_trainer.extract_embeddings(test_data_list)
    print(f"embeddings: {train_embeddings.shape}, {test_embeddings.shape}")
    # Label 준비 (각 stroke마다 동일한 skill level)
    train_labels_expanded = []
    for i, data in enumerate(train_data_list):
        train_labels_expanded.extend([train_labels[i]] * len(data))
    train_labels_expanded = np.array(train_labels_expanded)
    
    test_labels_expanded = []
    for i, data in enumerate(test_data_list):
        test_labels_expanded.extend([test_labels[i]] * len(data))
    test_labels_expanded = np.array(test_labels_expanded)
    
    print(f"Train embeddings: {train_embeddings.shape}")
    print(f"Test embeddings: {test_embeddings.shape}")
    
    # Skill Level 분류기 학습
    print("\n Training Skill Level Classifier ")
    for epoch in range(50):
        loss = skill_trainer.train_epoch(train_embeddings, train_labels_expanded, batch_size=64)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/50, Loss: {loss:.4f}')
    
    # 평가
    print("\n=== Evaluation ===")
    results = skill_trainer.evaluate(test_embeddings, test_labels_expanded)
    
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['report'])
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    
    # 피험자별 평가
    print("\n=== Per-Subject Evaluation ===")
    start_idx = 0
    for i, (subject, data) in enumerate(zip(test_subjects, test_data_list)):
        end_idx = start_idx + len(data)
        subject_preds = results['predictions'][start_idx:end_idx]
        subject_true = test_labels[i]
        
        # 다수결 투표
        from collections import Counter
        pred_counts = Counter(subject_preds)
        final_pred = pred_counts.most_common(1)[0][0]
        
        print(f"{subject}: True={subject_true}, Predicted={final_pred}, "
              f"Correct={final_pred == subject_true}")
        
        start_idx = end_idx
if __name__ == '__main__':
    main()