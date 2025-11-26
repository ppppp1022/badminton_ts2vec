import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
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

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth, num_classes, dropout=0.2):
        super(MLP, self).__init__()

        layers = []

        # ---- 1. 첫 번째 입력층 ----
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # ---- 2. 그 다음 depth-1 만큼 은닉층 반복 ----
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # ---- 3. 마지막 출력층 ----
        layers.append(nn.Linear(hidden_dim, num_classes))

        # Sequential로 묶기
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch, seq_len, feature) → BadmintonDataset에서 flatten되어 들어온다고 가정
        # 혹은 (batch, feature) 단일 벡터라면 그대로 통과됨

        # 만약 x가 (batch, seq_len, feature) 형태라면 flatten 필요:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        return self.network(x)
   

def run_kfold_experiment(dataset, stroke_type, joint_type, body_part, k=5, epoch = 20, batch_size=16,
                         hidden_dim=32, depth = 5, num_classes=7, device='cuda', output_dir='./results'):

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
        input_dim = len(sample_data[0])*len(sample_data)
        print(f"Input dimension: {input_dim}")
        print(f"\n[Step 1] Training mlp...{len(train_data)}")

        train_data = np.array(train_data)
        test_data = np.array(test_data)

        
        # 1. DataLoader 생성
        train_dataset = BadmintonDataset(train_data, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        
        test_dataset = BadmintonDataset(test_data, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)

        model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, depth=depth, num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss() # 회귀(예측) 문제 가정
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        for epoch_ in range(epoch):
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
            
            if (epoch_+1)%10 == 0:
                print(f"Epoch {epoch_+1}, Loss: {total_loss / len(train_loader):.4f}")
        
        model.eval()
        correct = 0
        total = 0

        # 🔥 Confusion matrix를 위해 전체 예측/정답을 저장할 리스트
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
    std_accuracy = np.std(accumulated_accuracy) * 100
    
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
    output_dir = './results_mlp'
    
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
        ('clear', 'global', 'arm'),
        #('clear', 'local', 'arm'),
        ('clear', 'global', 'leg'),
        #('clear', 'local', 'leg'),
        
        # Drive - 전체
        ('drive', 'global', 'total'),
        #('drive', 'local', 'total'),
        
        # Drive - 부위별
        ('drive', 'global', 'arm'),
        #('drive', 'local', 'arm'),
        ('drive', 'global', 'leg'),
        #('drive', 'local', 'leg'),
    ]
    # 각 실험 실행
    for stroke_type, joint_type, body_part in experiments:
        try:
            run_kfold_experiment(dataset=dataset,stroke_type=stroke_type,joint_type=joint_type,body_part=body_part,k=5,device=device,output_dir=output_dir,
                epoch=100,batch_size=64, hidden_dim=128, depth=7, num_classes=7)
            
        except Exception as e:
            print(f"\nERROR in experiment {stroke_type}_{joint_type}_{body_part}: {e}")
            import traceback
            traceback.print_exc()
        
    print(f"\nAll results saved to {output_dir}")

if __name__ == '__main__':
    main()