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
from ts2vec_skill_classification import collate_fn_with_labels

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

import torch
import torch.nn as nn
import math

# 1. 순서 정보를 주입하는 모듈 (보통 그대로 복사해서 씁니다)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # 위치별로 고유한 값을 미리 계산
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # (Batch 차원 추가) -> (1, max_len, d_model)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (Batch, Seq_Len, d_model)
        # 입력 길이에 맞춰서 위치 정보를 더해줌
        return x + self.pe[:, :x.size(1), :]

# 2. 메인 Transformer 모델
class SwingTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=128, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        
        # (1) 입력 차원 변환 (예: 20개 관절 -> 128차원)
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # (2) 위치 정보 추가
        self.pos_encoder = PositionalEncoding(d_model)
        
        # (3) 트랜스포머 인코더 층 설정
        # batch_first=True를 써야 (Batch, Time, Feature) 모양 그대로 넣을 수 있음
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # (4) 최종 분류기
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x, mask=None):
        """
        x: (Batch, Time, Input_Dim)
        mask: (Batch, Time) -> True가 진짜 데이터, False가 패딩인 마스크 (Collate_fn에서 만든 것)
        """
        
        # 1. 임베딩 & 위치 인코딩
        x = self.input_proj(x) # (Batch, Time, d_model)
        x = self.pos_encoder(x)
        
        # 2. 마스크 처리 (중요!)
        # PyTorch Transformer는 'True'를 패딩(무시할 곳)으로 인식합니다.
        # 하지만 우리가 만든 mask는 'True'가 진짜 데이터입니다.
        # 따라서 반전(~)을 시켜줘야 합니다.
        if mask is not None:
            # mask: (Batch, Time) -> True(Data), False(Pad)
            # padding_mask: True(Pad), False(Data) 로 변환
            padding_mask = ~mask
        else:
            padding_mask = None
            
        # 3. 인코더 통과
        # src_key_padding_mask에 반전된 마스크를 넣습니다.
        output = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # 4. Pooling (시계열 요약)
        # 방법 A: 단순히 평균 내기 (Global Average Pooling)
        # 패딩된 부분(0)까지 평균에 들어가면 안 되므로, 마스크를 고려해서 평균을 내거나
        # 단순히 Max Pooling을 쓰는 게 편합니다.
        
        # 여기선 TS2Vec처럼 Max Pooling 사용 (가장 강한 특징 추출)
        # (Batch, Time, d_model) -> (Batch, d_model)
        output = output.transpose(1, 2) # (Batch, d_model, Time)
        output = torch.max(output, dim=2)[0] 
        
        # 5. 최종 분류
        logits = self.classifier(output)
        
        return logits

def run_kfold_experiment(dataset, stroke_type, joint_type, body_part, k=5, transformer_epoch=100, device='cuda', 
                        hidden_size=256, num_layers=3, output_dir='./results'):
    """
    K-Fold 교차 검증 실험 실행
    """
    os.makedirs(output_dir, exist_ok=True)

    folds, labels = dataset.split_data_Kfold_randomly(stroke_type, k, body_part)
    accumulated_accuracy = []
    total_error = []

    for fold_idx in range(k):
        print(f"\n=== Fold {fold_idx+1}/{k} Start ===")
        
        # Fold에 따른 Subject 리스트 생성
        train_data = [stroke for i, fold in enumerate(folds) if i != fold_idx for stroke in fold]
        test_data = folds[fold_idx]
        train_labels =[l for i, label in enumerate(labels) if i != fold_idx for l in label]
        test_labels = labels[fold_idx]

        sample_data = train_data[0]
        input_size = len(sample_data[0])
        print(f"Input size: {input_size}, Train samples: {len(train_data)}, Test samples: {len(test_data)}")
        
        temp_tensors = [torch.tensor(s, dtype=torch.float32) for s in train_data]
        train_data_tensor = nn.utils.rnn.pad_sequence(temp_tensors, batch_first=True, padding_value=0)
        train_label_tensor = torch.tensor(train_labels, dtype=torch.long)

        temp_tensors = [torch.tensor(s, dtype=torch.float32) for s in test_data]
        test_data_tensor = nn.utils.rnn.pad_sequence(temp_tensors, batch_first=True, padding_value=0)
        test_label_tensor = torch.tensor(test_labels, dtype=torch.long)

        # DataLoader 생성
        train_dataset = TensorDataset(train_data_tensor, train_label_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_fn_with_labels)

        # [수정 1] 모델 정의: num_classes=7 (1~7단계를 분류하므로)
        # [수정 2] .to(device) 추가: 모델을 GPU로 이동
        model = SwingTransformer(input_size, 7).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # --- 학습 루프 ---
        print("Training lstm...")
        model.train()

        for epoch in range(transformer_epoch):
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
                print(f"Epoch [{epoch+1}/{transformer_epoch}], Avg Loss: {avg_loss:.4f}")

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
    output_dir = './results_transformer'
    
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
                output_dir=output_dir
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