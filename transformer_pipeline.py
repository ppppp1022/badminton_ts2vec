import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import matplotlib.pyplot as plt
import os
from preprocess_badminton_data import load_skill_levels_from_annotation
from processed_data_loader import ProcessedBadmintonDataset
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

class PositionalEncoding(nn.Module):
    """
    논문의 3.5절에 나오는 Positional Encoding 구현 [cite: 161]
    순서 정보(Time)를 주입하기 위해 사인/코사인 함수를 사용합니다.
    """
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # (Max_Len, d_model) -> (1, Max_Len, d_model)로 배치 차원 추가
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (Batch, Seq_Len, d_model)
        # 입력 길이에 맞춰서 잘라서 더해줌
        return x + self.pe[:, :x.size(1)]

class BadmintonTransformer(nn.Module):
    def __init__(self, input_dim=63, d_model=128, nhead=4, num_layers=3, num_classes=7, dropout=0.1):
        super(BadmintonTransformer, self).__init__()
        
        # 1. Input Projection (63 -> d_model)
        # 연속된 센서 값은 Embedding 레이어 대신 Linear 레이어를 씁니다.
        self.input_linear = nn.Linear(input_dim, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. Transformer Encoder Layer
        # 논문의 Encoder 구조 
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, Time, 63)
        
        # [Step 1] Feature Projection
        x = self.input_linear(x) # (Batch, Time, d_model)
        
        # [Step 2] Positional Encoding (순서 정보 주입)
        x = self.pos_encoder(x)
        
        # [Step 3] Transformer Encoder 통과
        # output shape: (Batch, Time, d_model)
        # 각 프레임이 전체 시퀀스를 참고한 특징을 갖게 됨
        x = self.transformer_encoder(x)
        
        # [Step 4] Aggregation (Global Average Pooling)
        # 모든 시간(Time) 축에 대해 평균을 내어 하나의 벡터로 만듦
        x = x.mean(dim=1) # (Batch, d_model)
        
        # [Step 5] Classification
        output = self.classifier(x)
        
        return output

class BadmintonDataset(Dataset):
    def __init__(self, data_list, labels):
        self.data = data_list
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 데이터: (Time, 63) 형태라고 가정
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def pad_collate_fn(batch):
    # batch = [(x1, y1), (x2, y2), ...]
    inputs, labels = zip(*batch)
    
    # 패딩 적용 (길이가 짧은 스윙 뒤에 0을 채움)
    # batch_first=True -> (Batch, Time, Feature)
    inputs_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    
    # 라벨 합치기
    labels = torch.tensor(labels, dtype=torch.long)
    return inputs_padded, labels

def run_kfold_experiment(dataset, stroke_type, joint_type, body_part, k=5, epoch = 20, batch_size = 64,
                         input_dim = 63, d_model = 128, nhead = 4, num_layer = 3, device='cuda', output_dir='./results'):

    os.makedirs(output_dir, exist_ok=True)

    folds, labels = dataset.split_data_Kfold_randomly(stroke_type, k, body_part)
    accumulated_accuracy = []
    cms = []

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
        test_data = np.array(test_data)

        
        # 1. DataLoader 생성
        train_dataset = BadmintonDataset(train_data, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        
        test_dataset = BadmintonDataset(test_data, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)

        model = BadmintonTransformer(input_dim=input_dim, d_model=d_model, nhead=nhead, num_layers=num_layer).to(device)
        
        criterion = nn.CrossEntropyLoss() # 회귀(예측) 문제 가정
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        for epoch in range(epoch):
            total_loss = 0
            for x_batch, y_batch in train_loader:
                # x_batch: (Batch, Seq_Len, Dims) -> Wrapper 내부에서 5D로 변환됨
                # y_batch: (Batch, Dims)
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                optimizer.zero_grad()
                
                pred = model(x_batch) # Forward
                
                loss = criterion(pred, y_batch) # Loss 계산
                loss.backward() # Backprop
                optimizer.step() # Update
                
                total_loss += loss.item()
            
            if (epoch+1)%10 == 0:
                print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")
        
        model.eval()
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, test_labels in test_loader:
                inputs, test_labels = inputs.to(device), test_labels.to(device)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                total += test_labels.size(0)
                correct += (predicted == test_labels).sum().item()

                # 🔥 CPU로 옮겨서 리스트에 저장
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(test_labels.cpu().numpy())
        accuracy = correct / total
        accumulated_accuracy.append(accuracy)
        # 🔥 혼동 행렬 계산
        cm = confusion_matrix(all_labels, all_preds)
        cms.append(cm.tolist())
            
    average_accuracy = sum(accumulated_accuracy) / k * 100
    std_accuracy = np.std(accumulated_accuracy)
    
    print(f"\n{'='*30}")
    print(f"Final Result ({k}-Fold CV)")
    print(f"Avg Accuracy: {average_accuracy:.4f}")
    print(f"{'='*30}")

    result_summary = {
        'experiment': f"{stroke_type}_{joint_type}_{body_part}_kfold",
        'fold_accuracies': round(average_accuracy, 2), # 스칼라 값 저장
        'fold_std': round(std_accuracy, 3),
        'confusion_matrix': cms,
        'fold_accuracies_list': accumulated_accuracy, # 상세 기록용 리스트도 저장 추천
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
    output_dir = './results_transformer'
    
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
        #('drive', 'global', 'leg'),
        #('drive', 'local', 'arm'),
        #('drive', 'local', 'leg'),
    ]
    # 각 실험 실행
    for stroke_type, joint_type, body_part in experiments:
        try:
            run_kfold_experiment(dataset=dataset,stroke_type=stroke_type,joint_type=joint_type,body_part=body_part,k=5,device=device,output_dir=output_dir,
                epoch=1,batch_size=64, input_dim = 63, d_model = 128, nhead = 4, num_layer = 3)
            
        except Exception as e:
            print(f"\nERROR in experiment {stroke_type}_{joint_type}_{body_part}: {e}")
            import traceback
            traceback.print_exc()
        
    print(f"\nAll results saved to {output_dir}")


if __name__ == '__main__':
    main()

