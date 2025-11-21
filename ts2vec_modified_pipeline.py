import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from preprocess_badminton_data import load_skill_levels_from_annotation
from processed_data_loader import ProcessedBadmintonDataset
import torch.optim as optim
import json
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def setup_dataset(processed_data_folder):
    """
    데이터셋 설정
    
    Args:
        annotation_filepath: Annotation Excel 파일 경로
        processed_data_folder: 전처리된 데이터 폴더 경로
    
    Returns:
        dataset: 설정된 ProcessedBadmintonDataset 객체
    """
    # 1. Annotation에서 skill level 로드
    print("Step 1: Loading skill levels from annotation file")
    
    
    clear_skills, drive_skills, subject_groups = load_skill_levels_from_annotation(annotation_filepath='./configs/skill_levels.json')

    # 2. 데이터셋 초기화
    print("Step 2: Initializing dataset")
    
    dataset = ProcessedBadmintonDataset(processed_data_folder)
    
    # Skill level 설정
    
    dataset.clear_skill_level = clear_skills
    dataset.drive_skill_level = drive_skills
    
    # 그룹 설정
    dataset.beginner_subjects = subject_groups['beginner']
    dataset.intermediate_subjects = subject_groups['intermediate']
    dataset.expert_subjects = subject_groups['expert']
    
    # 부위별 인덱스 설정 (사용자가 원하는 대로 수정)
    # Joint indices: 0=Hips, 1-6=Legs, 7-12=Spine/Neck/Head, 13-20=Arms
    dataset.local_arm_index = [13, 14, 15, 16, 17, 18, 19, 20]  # Right/Left Shoulder, Arm, ForeArm, Hand
    dataset.global_arm_index = [13, 14, 15, 16, 17, 18, 19, 20]
    
    dataset.local_leg_index = [0, 1, 2, 3, 4, 5, 6]  # Hips + Legs
    dataset.global_leg_index = [0, 1, 2, 3, 4, 5, 6]
    
    dataset.local_total_index = list(range(21))  # All joints
    dataset.global_total_index = list(range(21))
    
    # 통계 출력
    #dataset.get_statistics()
    
    return dataset

class DilatedConvEncoder(nn.Module):
    """
    논문에 기술된 Dilated CNN 아키텍처입니다.
    입력 투영(Input Projection) -> 타임스탬프 마스킹(Masking) -> Dilated CNN 순서로 처리합니다.
    """
    def __init__(self, in_channels, out_channels, hidden_size=64, depth=8):
        super().__init__()
        self.input_fc = nn.Linear(in_channels, hidden_size) # Input Projection Layer [cite: 86]
        
        # Dilated Convolutions (Residual Blocks)
        self.net = nn.Sequential()
        for i in range(depth):
            dilation = 2 ** i  # Dilation parameter 2^i [cite: 91]
            # 패딩을 사용하여 시간 축 길이를 유지합니다 (padding='same' 방식 구현)
            self.net.add_module(f'res_block_{i}', 
                                ResidualBlock(hidden_size, hidden_size, dilation))
            
        self.repr_dropout = nn.Dropout(p=0.1)
        self.out_fc = nn.Linear(hidden_size, out_channels)

    def forward(self, x, mask=None):
        # x shape: (Batch, Time, Features)
        x = self.input_fc(x)
        
        # Timestamp Masking [cite: 131, 132]
        # 마스크가 제공되면 잠재 벡터(latent vector)에 적용합니다.
        if mask is not None: 
            x = x * mask
            
        x = x.transpose(1, 2)  # Conv1d를 위해 (Batch, Channel, Time)으로 변경
        x = self.net(x)
        x = x.transpose(1, 2)  # 다시 (Batch, Time, Channel)로 복구
        
        return self.out_fc(x)

class ResidualBlock(nn.Module):
    """Dilated CNN을 위한 기본 블록"""
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + x) # Residual connection

def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    """
    계층적 대조 손실 (Hierarchical Contrastive Loss)[cite: 140, 141].
    Algorithm 1에 따라 시간 축으로 Max Pooling하며 반복 계산합니다.
    """
    loss = torch.tensor(0., device=z1.device)
    d = 0
    
    while z1.size(1) > 1:
        # Instance-wise & Temporal Contrastive Loss 계산 [cite: 155, 164]
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if alpha != 1:
            loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
            
        d += 1
        
        # 시간 축(dim=1)에 대해 Max Pooling 적용 (커널 크기 2)
        if z1.size(1) % 2 == 1:
            z1 = z1[:, :-1, :]
            z2 = z2[:, :-1, :]
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
        
    return loss / d


def instance_contrastive_loss(z1, z2):
    """
    [최적화] Instance-wise Contrastive Loss (Vectorized)
    - for 문 제거로 속도 대폭 향상
    """
    B, T = z1.size(0), z1.size(1)
    if B == 1: return z1.new_tensor(0.)
    
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    
    # 대각 성분 제외 로직 (Vectorized)
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    
    logits = -F.log_softmax(logits, dim=-1) # T x 2B x (2B-1)
    
    # --- [Vectorization 핵심] ---
    # for t in range(T) 루프를 없애고, T차원과 2B차원을 합칩니다.
    # (T * 2B, 2B - 1) 형태로 만듭니다.
    logits = logits.reshape(-1, 2 * B - 1)
    
    # 정답 타겟 생성 (한 타임스텝에 대한 정답)
    # z1 파트 (0~B-1) -> 정답: i + B - 1
    # z2 파트 (B~2B-1) -> 정답: i
    target_z1 = torch.arange(B, device=z1.device) + B - 1
    target_z2 = torch.arange(B, device=z1.device)
    target = torch.cat([target_z1, target_z2]) # (2B,)
    
    # 모든 타임스텝(T)에 대해 정답이 똑같으므로 반복해서 확장
    target = target.repeat(T) # (T * 2B,)
    
    # 한 번에 Gather & Mean (GPU 병렬 연산)
    loss = logits.gather(1, target.view(-1, 1)).mean()
    
    return loss

def temporal_contrastive_loss(z1, z2):
    """
    [최적화] Temporal Contrastive Loss (Vectorized)
    - for 문 제거로 속도 대폭 향상
    """
    B, T = z1.size(0), z1.size(1)
    if T == 1: return z1.new_tensor(0.)
    
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    
    logits = -F.log_softmax(logits, dim=-1) # B x 2T x (2T-1)
    
    # --- [Vectorization 핵심] ---
    # for b in range(B) 루프 제거
    # (B * 2T, 2T - 1) 형태로 Flatten
    logits = logits.reshape(-1, 2 * T - 1)
    
    # 정답 타겟 생성
    target_t1 = torch.arange(T, device=z1.device) + T - 1
    target_t2 = torch.arange(T, device=z1.device)
    target = torch.cat([target_t1, target_t2]) # (2T,)
    
    # 모든 배치(B)에 대해 정답 반복
    target = target.repeat(B) # (B * 2T,)
    
    loss = logits.gather(1, target.view(-1, 1)).mean()
    
    return loss

class TS2Vec:
    """사용자가 데이터를 넣기 쉽도록 만든 래퍼 클래스"""
    def __init__(self, input_dims, output_dims=320, hidden_dims=64, depth=10, device='cuda'):
        self.device = device
        self.hidden_dims = hidden_dims
        self.model = DilatedConvEncoder(input_dims, output_dims, hidden_dims, depth).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        
    def fit(self, train_data, n_epochs=20, batch_size=8, alpha = 0.3):
        """
        학습 함수.
        train_data shape: (N_samples, Time_length, Features)
        """
        self.model.train()
        
        N, T, F_dim = train_data.shape
        scaler = StandardScaler()
        input_data_scaled = scaler.fit_transform(train_data.reshape(-1, F_dim)).reshape(N, T, F_dim)
        
        train_tensor = torch.from_numpy(input_data_scaled).float().to(self.device)
        
        for epoch in range(n_epochs):
            idx = np.random.permutation(N)
            epoch_loss = 0
            
            for i in range(0, N, batch_size):
                batch = train_tensor[idx[i:i+batch_size]]
                
                # Random Cropping: 두 개의 겹치는 구간 샘플링 [cite: 135]
                ts_l = batch.size(1)
                crop_l = np.random.randint(low=2, high=ts_l + 1)
                
                # 논문 방식대로 간단한 랜덤 크롭 구현
                start1 = np.random.randint(ts_l - crop_l + 1)
                start2 = np.random.randint(ts_l - crop_l + 1)
                
                x1 = batch[:, start1 : start1 + crop_l, :]
                x2 = batch[:, start2 : start2 + crop_l, :]
                
                self.optimizer.zero_grad()
                
                # Timestamp Masking 생성 (Bernoulli p=0.5) [cite: 132]
                
                mask1 = torch.from_numpy(np.random.binomial(1, 0.2, size=(x1.shape[0], x1.shape[1], 1))).to(self.device).float()
                mask2 = torch.from_numpy(np.random.binomial(1, 0.2, size=(x2.shape[0], x2.shape[1], 1))).to(self.device).float()
                
                # Forward Pass (두 개의 뷰 생성)
                z1 = self.model(x1, mask1)
                z2 = self.model(x2, mask2)
                
                # Hierarchical Loss 계산
                loss = hierarchical_contrastive_loss(z1, z2, alpha=alpha)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs} | Loss: {epoch_loss / (N//batch_size):.4f}")

    
    def encode(self, data, batch_size=8):
        """
        추론 함수. 전체 시계열에 대한 표현을 반환합니다.
        Classification 등의 task에 사용 시 Max Pooling을 적용하여 Instance level 표현을 얻습니다.
        """
        self.model.eval()
        data_tensor = torch.from_numpy(data).float().to(self.device)
        outputs = []
        
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = data_tensor[i:i+batch_size]
                out = self.model(batch, mask=None) # 추론 시에는 마스크 없음 [cite: 138]
                
                # Instance-level 표현을 원하면 여기서 Max Pooling 수행 [cite: 177]
                # out = F.max_pool1d(out.transpose(1, 2), kernel_size=out.size(1)).squeeze(-1)
                
                outputs.append(out.cpu().numpy())
                
        return np.concatenate(outputs, axis=0)

def visualize_embeddings(embeddings, labels, fold_idx, save_dir, suffix="Test"):
    """
    T-SNE 시각화 및 저장
    Args:
        embeddings: 임베딩 벡터 (numpy array)
        labels: 라벨 (numpy array)
        fold_idx: 현재 Fold 번호
        save_dir: 저장할 폴더 경로 (예: ./results/tsne_images)
        suffix: 파일명 및 제목 접미사 (Train 또는 Test)
    """
    embeddings = np.max(embeddings, axis=1)
    print(f"Pooling 후 형태: {embeddings.shape}") # (3000, 64)

    if type(labels) is list:
        labels = np.array(labels)

    # NaN 값 안전 장치 (이전 에러 방지)
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        print(f"⚠️ Warning: NaN found in {suffix} embeddings. Replacing with 0 for visualization.")
        embeddings = np.nan_to_num(embeddings)
        embeddings[np.isinf(embeddings)] = 0

    print(f"Generating T-SNE for Fold {fold_idx} ({suffix})...")
    
    # T-SNE 설정 (init='random'으로 안정성 확보)
    tsne = TSNE(
        n_components=2, 
        random_state=42, 
        perplexity=30, 
        max_iter=1000,
        init='random',
        learning_rate='auto'
    )
    
    # 데이터가 너무 많으면 시간이 오래 걸리므로 3000개 정도로 샘플링하는 것도 방법
    # 여기선 일단 전체 다 그립니다.
    tsne_results = tsne.fit_transform(embeddings)

    print(f"t-SNE 결과 형태: {tsne_results.shape}") # (3000, 2)
    # 시각화 데이터프레임
    
    df = pd.DataFrame({
        'x': tsne_results[:, 0],
        'y': tsne_results[:, 1],
        'Skill Level': labels.astype(int)
    })
    
    # 그래프 그리기
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df, 
        x='x', y='y', 
        hue='Skill Level', 
        palette='viridis', 
        style='Skill Level',
        s=50, 
        alpha=0.7
    )
    
    plt.title(f'Fold {fold_idx} T-SNE ({suffix})')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    
    # 파일 저장 (폴더는 이미 만들어져 있다고 가정하거나 여기서 생성)
    os.makedirs(save_dir, exist_ok=True)
    filename = f'tsne_fold_{fold_idx}_{suffix}.png'
    save_path = os.path.join(save_dir, filename)
    
    plt.savefig(save_path, dpi=300)
    plt.close() # 메모리 해제
    print(f"Saved: {save_path}")


class TS2VecNNClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, dropout=0.2, device='cuda'):
        super().__init__()
        self.device = device
        
        # --- 여기가 진짜 Neural Network (MLP) 구조입니다 ---
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),      # 입력 -> 은닉층
            nn.BatchNorm1d(hidden_dim),            # 학습 안정화 (선택 사항)
            nn.ReLU(),                             # 비선형 활성화 함수 (이게 있어야 NN입니다)
            nn.Dropout(p=dropout),                 # 과적합 방지
            nn.Linear(hidden_dim, num_classes)     # 은닉층 -> 최종 클래스 개수
        ).to(device)
        
        # CrossEntropyLoss가 내부적으로 Softmax를 수행하고 정수 라벨과 비교합니다.
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.parameters(), lr=0.001)

    def _pool(self, x):
        """(N, T, F) -> (N, F) Max Pooling"""
        if x.ndim == 3:
            x = torch.max(x, dim=1)[0]
        return x

    def fit(self, train_embeddings, train_labels, epochs=100, batch_size=64, lr=0.001):
        self.train()
        
        # Optimizer LR 업데이트
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        # Numpy -> Tensor 변환
        X_tensor = torch.from_numpy(train_embeddings).float().to(self.device)
        y_tensor = torch.from_numpy(train_labels).long().to(self.device) # 라벨은 반드시 long(정수)이어야 함
        
        # Max Pooling으로 시계열 압축
        X_tensor = self._pool(X_tensor)
        
        N = X_tensor.size(0)
        loss_history = []
        
        print(f"=== Neural Network 학습 시작 (Epochs: {epochs}) ===")
        
        for epoch in range(epochs):
            idx = torch.randperm(N)
            X_shuffled = X_tensor[idx]
            y_shuffled = y_tensor[idx]
            
            epoch_loss = 0
            correct = 0
            total = 0
            
            for i in range(0, N, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                self.optimizer.zero_grad()
                
                # 1. 모델 예측 (Logits 출력: 실수형 값들)
                outputs = self.classifier(X_batch)
                
                # 2. Loss 계산 (여기서 정수형 라벨과 비교하여 학습됨)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                # 3. 정확도 계산 (가장 높은 값을 가진 인덱스를 정답으로 선택)
                _, predicted = torch.max(outputs.data, 1) # 여기서 정수형 예측값이 나옴
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                
            avg_loss = epoch_loss / (N / batch_size)
            acc = 100 * correct / total
            loss_history.append(avg_loss)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
                
        print("=== 학습 완료 ===")

    def evaluate(self, test_embeddings, test_labels, label_names=None):
        self.eval()
        X_tensor = torch.from_numpy(test_embeddings).float().to(self.device)
        
        # Max Pooling
        X_tensor = self._pool(X_tensor)
        
        with torch.no_grad():
            outputs = self.classifier(X_tensor)
            # outputs는 [0.1, 0.8, 0.1] 같은 확률 분포와 유사한 값
            # torch.max(..., 1)을 하면 가장 큰 값의 '인덱스(정수)'를 반환함 -> 이것이 분류 결과
            _, predicted = torch.max(outputs.data, 1)
            
        y_pred = predicted.cpu().numpy()
        y_true = test_labels
        
        acc = accuracy_score(y_true, y_pred)
        print(f"\n>>> 최종 테스트 정확도: {acc:.4f} ({acc*100:.2f}%) <<<\n")
        
        print(">>> 상세 분류 리포트 <<<")
        log = classification_report(y_true, y_pred, target_names=label_names, digits=4)
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        
        return acc, log

def run_kfold_experiment(dataset, stroke_type, joint_type, body_part, k=5, epoch = 20, batch_size=16,
                         hidden_dim = 64, output_dim = 64, alpha = 0.3, device='cuda', output_dir='./results'):
    """
    K-Fold 교차 검증 실험 실행 함수.
    
    Args:
        dataset: ProcessedBadmintonDataset 객체
        stroke_type: 'clear' 또는 'drive'
        joint_type: 'global' 또는 'local'
        body_part: 'total', 'arm', 또는 'leg'
        k: K-Fold 개수
        epoch: TS2Vec 학습 epoch 수
        output_dim: TS2Vec 출력 차원
        hidden_dim: TS2Vec 은닉 차원
        batch_size: 배치 크기
        device: 학습에 사용할 디바이스 ('cuda', 'mps', 'cpu' 등)
        output_dir: 결과 저장 디렉토리 경로
    Returns:
        results: 실험 결과 딕셔너리
    """
    os.makedirs(output_dir, exist_ok=True)
    
    tsne_save_dir = os.path.join(output_dir, f'tsne_images_{batch_size}_{hidden_dim}_{output_dim}_{alpha}')
    os.makedirs(tsne_save_dir, exist_ok=True)

    folds, labels = dataset.split_data_Kfold_randomly(stroke_type, k, body_part)
    accumulated_accuracy = []
    logs = []

    print(f"Starting experiment: {stroke_type}_{joint_type}_{body_part}")
    
    for fold_idx in range(k):
        print(f"\n{'='*20}\n Fold {fold_idx+1} / {k} \n{'='*20}")
        
        # --- 1. 데이터 준비 ---
        train_data = [stroke for i, fold in enumerate(folds) if i != fold_idx for stroke in fold]
        test_data = folds[fold_idx]
        train_labels =[l-1 for i, label in enumerate(labels) if i != fold_idx for l in label]
        test_labels = [l-1 for i, label in enumerate(labels) if i == fold_idx for l in label]

        sample_data = train_data[0]
        input_dim = len(sample_data[0]) 
        print(f"Input dimension: {input_dim}")
        print(f"\n[Step 1] Training TS2Vec (Self-Supervised)...{len(train_data)}")

        train_data = np.array(train_data)
        velocity_data = np.diff(train_data, axis=1)
        acceleration_data = np.diff(velocity_data, axis=1)
        test_data = np.array(test_data)

        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)

        model = TS2Vec(input_dims=input_dim, hidden_dims=hidden_dim, output_dims=output_dim, device=device)
        model.fit(train_data=train_data, n_epochs=epoch, batch_size=batch_size)

        train_embeddings = model.encode(train_data)
        test_embeddings = model.encode(test_data)

        visualize_embeddings(
            train_embeddings, 
            train_labels, 
            fold_idx=fold_idx+1, 
            save_dir=tsne_save_dir, # 하위 폴더 지정
            suffix="Train"
        )

        nn_classifier = TS2VecNNClassifier(input_dim=output_dim, num_classes=7, hidden_dim=128, device=device)
        nn_classifier.fit(train_embeddings, train_labels, epochs=100)

        accuracy, log = nn_classifier.evaluate(test_embeddings, test_labels)

        accumulated_accuracy.append(accuracy)
        logs.append(log)
    
    average_accuracy = sum(accumulated_accuracy) / k
    
    print(f"\n{'='*30}")
    print(f"Final Result ({k}-Fold CV)")
    print(f"Avg Accuracy: {average_accuracy:.4f}")
    print(f"{'='*30}")

    result_summary = {
        'experiment': f"{stroke_type}_{joint_type}_{body_part}_kfold",
        'parameter': f"{epoch}, {batch_size}, {hidden_dim}, {output_dim}, {alpha}",
        'fold_accuracies': average_accuracy, # 스칼라 값 저장
        'fold_accuracies_list': accumulated_accuracy, # 상세 기록용 리스트도 저장 추천
        'log': logs
    }

    # JSON 누적 저장 로직
    summary_file_path = os.path.join(output_dir, 'all_results_summary.json')
    
    existing_data = []
    if os.path.exists(summary_file_path):
        with open(summary_file_path, 'r', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
            except json.JSONDecodeError:
                existing_data = []
                
    existing_data.append(result_summary)

    with open(summary_file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)

    print(f"All results saved to {summary_file_path}")
    return result_summary

def main():
    processed_data_folder = './Processed_Data'
    output_dir = './results_ts2vec'
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 데이터셋 설정
    dataset = setup_dataset(processed_data_folder)
    
    # 실험 조합 정의
    experiments = [
        # Clear - 전체
        ('clear', 'global', 'total'),
        #('clear', 'local', 'total'),
        
        # Clear - 부위별
        #('clear', 'global', 'arm'),
        #('clear', 'local', 'arm'),
        #('clear', 'global', 'leg'),
        #('clear', 'local', 'leg'),
        
        # Drive - 전체
        #('drive', 'global', 'total'),
        #('drive', 'local', 'total'),
        
        # Drive - 부위별
        #('drive', 'global', 'arm'),
        #('drive', 'local', 'arm'),
        #('drive', 'global', 'leg'),
        #('drive', 'local', 'leg'),
    ]
    # 각 실험 실행
    for stroke_type, joint_type, body_part in experiments:
        try:
            run_kfold_experiment(dataset=dataset,stroke_type=stroke_type,joint_type=joint_type,body_part=body_part,k=5,device=device,output_dir=output_dir,
                epoch=150,batch_size=64,hidden_dim=128,output_dim=256,alpha = 0.3)
            run_kfold_experiment(dataset=dataset,stroke_type=stroke_type,joint_type=joint_type,body_part=body_part,k=5,device=device,output_dir=output_dir,
                epoch=100,batch_size=64,hidden_dim=128,output_dim=256,alpha = 0.5)
            run_kfold_experiment(dataset=dataset,stroke_type=stroke_type,joint_type=joint_type,body_part=body_part,k=5,device=device,output_dir=output_dir,
                epoch=100,batch_size=64,hidden_dim=128,output_dim=256,alpha = 0.7)
            run_kfold_experiment(dataset=dataset,stroke_type=stroke_type,joint_type=joint_type,body_part=body_part,k=5,device=device,output_dir=output_dir,
                epoch=100,batch_size=64,hidden_dim=128,output_dim=512,alpha = 0.3)
        except Exception as e:
            print(f"\nERROR in experiment {stroke_type}_{joint_type}_{body_part}: {e}")
            import traceback
            traceback.print_exc()
        
    print(f"\nAll results saved to {output_dir}")


if __name__ == '__main__':
    main()