import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from preprocess_badminton_data import load_skill_levels_from_annotation
from processed_data_loader import ProcessedBadmintonDataset
import torch.optim as optim
import json
import os
import numpy as np

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

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
        super(LSTMClassifier, self).__init__()
        
        # batch_first=True 필수
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # 마지막 타임스텝의 결과를 분류
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x: (Batch, Time, 63)
        output, (hn, cn) = self.lstm(x)
        
        # output[:, -1, :] : (Batch, Time, Hidden) 중 마지막 Time의 Hidden만 가져옴
        # "스윙이 다 끝난 뒤의 요약 정보"를 사용
        out = output[:, -1, :] 
        
        return self.fc(out)

def pad_collate_fn(batch):
    # batch = [(x1, y1), (x2, y2), ...]
    inputs, labels = zip(*batch)
    
    # 패딩 적용 (길이가 짧은 스윙 뒤에 0을 채움)
    # batch_first=True -> (Batch, Time, Feature)
    inputs_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    
    # 라벨 합치기
    labels = torch.tensor(labels, dtype=torch.long)
    
    return inputs_padded, labels

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

def collate_fn_with_labels(batch):
    
    padded = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    print(batch)
    
    return padded

def train_lstm_fold(train_data, train_labels, test_data, test_labels, input_size=63, num_classes=4, device='cuda'):
    
    # --- 하이퍼파라미터 설정 ---
    BATCH_SIZE = 64
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    LEARNING_RATE = 0.001
    EPOCHS = 100  # 적절히 조절하세요

    # 1. DataLoader 생성
    train_dataset = BadmintonDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate_fn)
    
    test_dataset = BadmintonDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn)

    # 2. 모델 초기화
    model = LSTMClassifier(input_size, HIDDEN_SIZE, num_classes, NUM_LAYERS).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. 학습 루프
    print(f"Start Training (Train: {len(train_data)}, Test: {len(test_data)})")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # 10 에폭마다 로그 출력
        if (epoch + 1) % 10 == 0:
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

    # 4. 최종 평가 (Test Set)
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Fold Final Accuracy: {acc:.2f}%")
    print("-" * 30)
    
    return acc

def run_kfold_experiment(dataset, stroke_type, joint_type, body_part, k=5, device='cuda', output_dir='./results'):
    import numpy as np

    def inspect_labels(name, labels):
        # 리스트일 수도 있고 텐서일 수도 있으니 안전하게 변환
        if hasattr(labels, 'cpu'): # 텐서라면
            arr = labels.cpu().numpy()
        else: # 리스트나 넘파이라면
            arr = np.array(labels)
            
        # 혹시 차원이 (N, 1) 처럼 되어있을까봐 1차원으로 쫙 폄
        arr = arr.flatten()
        
        unique_vals = np.unique(arr)
        
        print(f"=== [{name}] 분석 결과 ===")
        print(f"1. 데이터 개수: {len(arr)}")
        print(f"2. 고유값(종류): {unique_vals}")
        print(f"3. 최소값: {np.min(arr)}")
        print(f"4. 최대값: {np.max(arr)}")
        print(f"5. 클래스 개수 추정: {len(unique_vals)}")
        print("-" * 30)
    """
    K-Fold 교차 검증 실험 실행
    """
    os.makedirs(output_dir, exist_ok=True)

    folds, labels = dataset.split_data_Kfold_randomly(stroke_type, k, body_part)
    accumulated_accuracy = []

    for fold_idx in range(k):
        print(f"\n=== Fold {fold_idx+1}/{k} Start ===")
        
        # Fold에 따른 Subject 리스트 생성
        train_data = [stroke for i, fold in enumerate(folds) if i != fold_idx for stroke in fold]
        test_data = folds[fold_idx]
        train_labels =[l-1 for i, label in enumerate(labels) if i != fold_idx for l in label]
        test_labels = [l-1 for i, label in enumerate(labels) if i == fold_idx for l in label]
        sample_data = train_data[0]
        input_size = len(sample_data[0])
        print(f"Input size: {input_size}, Train samples: {len(train_data)}, Test samples: {len(test_data)}")
        
        # DataLoader 생성
        NUM_CLASSES = 7

        # 디바이스 설정
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 함수 호출!
        accuracy = train_lstm_fold(
            train_data, 
            train_labels, 
            test_data, 
            test_labels, 
            input_size=input_size,       # 로그에 찍힌 63
            num_classes=NUM_CLASSES, 
            device=device
        )
        accumulated_accuracy.append(accuracy)

    average_accuracy = np.mean(accumulated_accuracy)
    print(f"\nAverage Accuracy over {k} folds: {average_accuracy:.4f}")

    result_summary = {
        'experiment': f"{stroke_type}_{joint_type}_{body_part}_kfold",
        'fold_accuracies': accumulated_accuracy,
        'average_accuracy': average_accuracy
    }
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
    output_dir = './results_lstm'
    
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
    all_results = []
    for stroke_type, joint_type, body_part in experiments:
        try:
            result = run_kfold_experiment(
                dataset,
                stroke_type,
                joint_type,
                body_part,
                device=device,
                output_dir=output_dir
            )
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"\nERROR in experiment {stroke_type}_{joint_type}_{body_part}: {e}")
            import traceback
            traceback.print_exc()
        
    print(f"\nAll results saved to {output_dir}")


if __name__ == '__main__':
    main()