import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
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

class LSTMClassification(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        # 마지막 시점의 정보를 이용해 클래스 확률(점수) 계산
        scores = self.fc(hn[-1]) 
        return scores

def run_kfold_experiment(dataset, stroke_type, joint_type, body_part, k=5, lstm_epoch=100, device='cuda', 
                        hidden_size=256, num_layers=3, batch_first=True):
    """
    K-Fold 교차 검증 실험 실행
    """
    
    # 데이터셋에서 subject 분할
    folds, labels = dataset.split_data_Kfold(stroke_type, k)
    accumulated_accuracy = []
    mean_errors = []
    
    for fold_idx in range(k):
        print(f"\n=== Fold {fold_idx+1}/{k} Start ===")
        
        # Fold에 따른 Subject 리스트 생성
        train_subjects = [s for i, fold in enumerate(folds) if i != fold_idx for s in fold]
        test_subjects = folds[fold_idx]

        # --- Train Data Loading ---
        train_sequences = [] 
        train_labels_list = [] # 변수명 명확히
        
        for subject in train_subjects:
            strokes, skill = dataset.load_subject_data(subject, stroke_type, joint_type, body_part)
            for single_stroke in strokes:
                seq_tensor = torch.tensor(single_stroke, dtype=torch.float32)
                train_sequences.append(seq_tensor)
                
                # [중요] Skill 1~7 -> Label 0~6 변환
                train_labels_list.append(skill - 1) 
        
        # --- Test Data Loading ---
        print("Loading test data...")
        test_sequences = []
        test_labels_list = []

        for subject in test_subjects:
            strokes, skill = dataset.load_subject_data(subject, stroke_type, joint_type, body_part)
            for single_stroke in strokes:
                seq_tensor = torch.tensor(single_stroke, dtype=torch.float32)
                test_sequences.append(seq_tensor)
                test_labels_list.append(skill - 1) # 여기도 똑같이 -1

        input_size = train_sequences[0].shape[1]
        print(f"Input size: {input_size}, Train samples: {len(train_sequences)}, Test samples: {len(test_sequences)}")
        
        # --- Tensor 변환 & Padding ---
        train_data_tensor = pad_sequence(train_sequences, batch_first=True, padding_value=0)
        train_label_tensor = torch.tensor(train_labels_list, dtype=torch.long)
        
        test_data_tensor = pad_sequence(test_sequences, batch_first=True, padding_value=0)
        test_label_tensor = torch.tensor(test_labels_list, dtype=torch.long)
        
        # DataLoader 생성
        train_dataset = TensorDataset(train_data_tensor, train_label_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # [수정 1] 모델 정의: num_classes=7 (1~7단계를 분류하므로)
        # [수정 2] .to(device) 추가: 모델을 GPU로 이동
        model = LSTMClassification(input_size=input_size, hidden_size=hidden_size, 
                                   num_layers=num_layers, num_classes=7).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # --- 학습 루프 ---
        print("Training lstm...")
        model.train()

        for epoch in range(lstm_epoch):
            epoch_loss = 0.0
            
            for inputs, labels in train_loader:
                # [수정 3] 데이터도 GPU로 이동해야 함
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()

            if (epoch+1) % 10 == 0:
                avg_loss = epoch_loss / len(train_loader)
                print(f"Epoch [{epoch+1}/{lstm_epoch}], Avg Loss: {avg_loss:.4f}")

        # --- 평가 루프 ---
        print("Evaluating model...")
        
        # [수정 4] Test 데이터 GPU 이동 및 변수명 통일 (test_label_tensor)
        test_data_tensor = test_data_tensor.to(device)
        test_label_tensor = test_label_tensor.to(device)
        
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_data_tensor)
            predicted_classes = torch.argmax(test_outputs, dim=1)
            
            # [수정 5] 정확도 계산 시 변수명 오타 수정
            correct_tensor = (predicted_classes == test_label_tensor)
            mean_errors.append(np.mean(abs(avg_pred - subject_true) for subject_true, avg_pred in zip(test_label_tensor.cpu().numpy(), predicted_classes.cpu().numpy())))
            correct_count = correct_tensor.sum().item()
            total_count = test_label_tensor.size(0)

            accuracy = correct_count / total_count
            print(f"Fold {fold_idx+1} Accuracy: {accuracy:.4f}")
        mean_errors = np.mean(mean_errors)
        accumulated_accuracy.append(accuracy)

    average_accuracy = sum(accumulated_accuracy) / k
    print(f"\nAverage Accuracy over {k} folds: {average_accuracy:.4f}")

    result_summary = {
        'experiment': f"{stroke_type}_{joint_type}_{body_part}_kfold",
        'fold_accuracies': accumulated_accuracy,
        'average_accuracy': average_accuracy,
        'error': mean_errors
    }
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
        ('clear', 'local', 'total'),
        
        # Clear - 부위별
        ('clear', 'global', 'arm'),
        ('clear', 'local', 'arm'),
        ('clear', 'global', 'leg'),
        ('clear', 'local', 'leg'),
        
        # Drive - 전체
        ('drive', 'global', 'total'),
        ('drive', 'local', 'total'),
        
        # Drive - 부위별
        ('drive', 'global', 'arm'),
        ('drive', 'local', 'arm'),
        ('drive', 'global', 'leg'),
        ('drive', 'local', 'leg'),
    ]
    """experiments = [
            # Clear - 전체
            ('clear', 'local', 'total')
    ]"""
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
                lstm_epoch=100,
                hidden_size=256,
                num_layers=3,
                batch_first=True
            )
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"\nERROR in experiment {stroke_type}_{joint_type}_{body_part}: {e}")
            import traceback
            traceback.print_exc()
        
    # 전체 결과 저장
    with open(os.path.join(output_dir, 'all_results_summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll results saved to {output_dir}")


if __name__ == '__main__':
    main()