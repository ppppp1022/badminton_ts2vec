import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from typing import List, Dict
import os
from processed_data_loader import ProcessedBadmintonDataset
from preprocess_badminton_data import load_skill_levels_from_annotation
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error

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

class AttentionSkillClassifier(nn.Module):
    """Self-attention 메커니즘 사용"""
    def __init__(self, embedding_dim, num_classes=7):
        super().__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x: (batch, embedding_dim)
        x_expanded = x.unsqueeze(1)  # (batch, 1, embedding_dim)
        
        # Self-attention
        attn_output, _ = self.attention(x_expanded, x_expanded, x_expanded)
        attn_output = attn_output.squeeze(1)  # (batch, embedding_dim)
        
        # Residual connection
        x = x + attn_output
        
        # Classification
        return self.classifier(x)

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

class TS2VecTrainer:
    """TS2Vec 학습 클래스"""
    def __init__(self, model, lr=0.001, temperature=0.05, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = TS2VecLoss(temperature=temperature)
        
    def augment(self, x):
        """
        TS2Vec-style augmentation (강화 버전)
        1. Jittering (노이즈): 강도 증가
        2. Scaling (크기): 범위 확대
        3. Permutation (구간 섞기): [New] 시간 순서를 섞어서 '지렁이' 끊기
        """
        batch_size, seq_len, feature_dim = x.shape
        
        # --- 1. Stronger Jittering (노이즈 추가) ---
        # 기존 0.005 -> 0.02~0.03으로 상향 (데이터 스케일에 따라 조절 필요)
        noise = torch.randn_like(x) * 0.02
        x_aug = x + noise
        
        # --- 2. Stronger Scaling (크기 변형) ---
        # 기존 0.1 -> 0.3으로 상향 (0.7배 ~ 1.3배 크기로 변함)
        scale = torch.randn(batch_size, 1, 1, device=x.device) * 0.3 + 1
        x_aug = x_aug * scale
        """
        # --- 3. Permutation (구간 섞기 - 핵심!) ---
        # 시계열을 N개의 구간으로 자른 뒤 순서를 랜덤하게 섞습니다.
        # 모델이 "전체적인 시간 흐름"보다는 "동작의 핵심 패턴"에 집중하게 만듭니다.
        if np.random.random() < 0.5:
            mask_len = x.shape[1] // 4  # 전체 길이의 1/4 정도를 가림
            start = np.random.randint(0, x.shape[1] - mask_len)
            x_aug[:, start:start+mask_len, :] = 0"""
        # 30%의 확률로만 적용 (너무 많이 하면 원본 정보가 파괴될 수 있음)
        """if np.random.random() < 0.3:
            max_segments = 5
            num_segments = np.random.randint(2, max_segments + 1)
            
            # 구간 길이 계산
            seg_len = seq_len // num_segments
            
            # 텐서를 구간별로 쪼갬
            segments = []
            for i in range(num_segments):
                start = i * seg_len
                # 마지막 구간은 남은거 다 포함
                end = (i + 1) * seg_len if i < num_segments - 1 else seq_len
                segments.append(x_aug[:, start:end, :])
                
            # 섞기 (Shuffle)
            import random
            random.shuffle(segments)
            
            # 다시 합치기
            x_aug = torch.cat(segments, dim=1)"""
            
        return x, x_aug # x: 원본(약한 aug), x_aug: 강한 aug
    
    def train_epoch(self, data_list: List[np.ndarray], batch_size=32):
        """1 에폭 학습"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # 1. 모든 stroke를 하나의 리스트로 평탄화 (Flatten)
        all_strokes = []
        for subject_data in data_list:
            for stroke in subject_data:
                # numpy array -> torch tensor 변환을 미리 해둠
                all_strokes.append(torch.tensor(stroke, dtype=torch.float32))
        
        # 2. Shuffle (리스트 자체를 섞음)
        # 길이가 다르므로 np.random.permutation 인덱싱 대신 리스트를 섞는게 안전함
        import random
        random.shuffle(all_strokes)

        # 3. Batch 학습 (직접 루프)
        for i in range(0, len(all_strokes), batch_size):
            batch_list = all_strokes[i:i+batch_size]
            
            if len(batch_list) < 2: # 배치가 너무 작으면 패스 (Contrastive Loss 계산 불가)
                continue
            
            # [핵심 수정] pad_sequence로 배치 내 길이 통일 (Padding)
            # 결과 shape: (Batch, Max_Length, Feature)
            batch_tensor = pad_sequence(batch_list, batch_first=True, padding_value=0).to(self.device)
            
            # [핵심 수정] Mask 생성 (Padding된 부분은 False, 진짜 데이터는 True)
            # (Batch, Max_Length) 형태
            mask = (batch_tensor.abs().sum(dim=-1) > 0)
            
            # Augmentation
            _, x1 = self.augment(batch_tensor) # 첫 번째 뷰
            _, x2 = self.augment(batch_tensor) # 두 번째 뷰
            
            # Forward (Mask 전달!)
            # 이전 턴의 TS2Vec 모델이 mask를 받도록 수정되었다고 가정
            _, z1 = self.model(x1, mask=mask)
            _, z2 = self.model(x2, mask=mask)
            
            # Loss Calculation
            loss = self.criterion(z1, z2)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0

    def save_model(self, stroke_type: str, joint_type: str, body_part: str, save_dir='./EmbeddingModel'):
        """임베딩 모델 저장"""
        os.makedirs(save_dir, exist_ok=True)
        
        model_name = f"ts2vec_{stroke_type}_{joint_type}_{body_part}.pth"
        model_path = os.path.join(save_dir, model_name)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'input_dim': self.model.input_dim,
                'hidden_dim': self.model.hidden_dim,
                'output_dim': self.model.output_dim
            },
            'stroke_type': stroke_type,
            'joint_type': joint_type
        }
        
        torch.save(checkpoint, model_path)
        print(f"Model saved to {model_path}")
        return model_path
    
    @staticmethod
    def load_model(stroke_type: str, joint_type: str, body_part: str, device='cuda', save_dir='./EmbeddingModel'):
        """저장된 임베딩 모델 로드"""
        model_name = f"ts2vec_{stroke_type}_{joint_type}_{body_part}.pth"
        model_path = os.path.join(save_dir, model_name)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=device)
        
        # 모델 재생성 부분에서 class TS2Vec이 정의되어 있어야 함
        config = checkpoint['model_config']
        
        model = TS2Vec(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        trainer = TS2VecTrainer(model, device=device)
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Model loaded from {model_path}")
        return model, trainer

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
        
    def extract_embeddings(self, data_list: List[np.ndarray], batch_size=64) -> np.ndarray:
        """
        TS2Vec으로 임베딩 추출 (Batch Processing + Padding/Masking 적용)
        """
        self.ts2vec_model.eval()
        embeddings = []
        
        # 1. 모든 데이터를 하나의 리스트로 풀기 (Flatten)
        all_strokes = []
        for subject_data in data_list:
            for stroke in subject_data:
                all_strokes.append(torch.tensor(stroke, dtype=torch.float32))
        
        # 2. 배치 단위로 처리 (속도 대폭 향상)
        with torch.no_grad():
            for i in range(0, len(all_strokes), batch_size):
                batch_list = all_strokes[i:i+batch_size]
                
                # [중요] Padding: 길이가 다른 스윙들을 맞춰줌
                batch_tensor = pad_sequence(batch_list, batch_first=True, padding_value=0).to(self.device)
                
                # [중요] Mask 생성: 0인 부분은 임베딩 계산에서 제외
                mask = (batch_tensor.abs().sum(dim=-1) > 0)
                
                # TS2Vec Encode (Mask 전달!)
                # shape: (Batch, Output_Dim)
                embedding_batch = self.ts2vec_model.encode(batch_tensor, mask=mask)
                
                embeddings.append(embedding_batch.cpu().numpy())
        
        # 리스트들을 하나의 거대한 넘파이 배열로 합침
        return np.vstack(embeddings)
    
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
# ============================================================================
# 5. 메인 실행 파이프라인
# ============================================================================

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