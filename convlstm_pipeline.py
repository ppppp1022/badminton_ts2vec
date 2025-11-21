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

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        """
        초기화 함수
        :param input_dim: 입력 이미지의 채널 수 (예: Radar map 채널)
        :param hidden_dim: 은닉 상태(Hidden State)의 채널 수
        :param kernel_size: 커널 크기 (예: (3,3) 또는 3)
        :param bias: 바이어스 사용 여부
        """
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 커널의 높이와 너비가 홀수여야 패딩을 통해 입력 크기를 유지하기 쉽습니다.
        # 논문에서는 상태(State)의 크기가 입력과 동일하게 유지되도록 패딩을 사용한다고 명시되어 있습니다[cite: 107].
        padding = kernel_size[0] // 2, kernel_size[1] //2

        # 합성곱 연산 정의 (Input + Hidden 상태를 채널 방향으로 합쳐서 연산)
        # 출력 채널은 4 * hidden_dim (Input gate, Forget gate, Cell gate, Output gate)
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)

    def forward(self, input_tensor, cur_state):
        """
        Forward Pass
        :param input_tensor: (Batch, Input_Channel, Height, Width) - 4D 텐서 [cite: 96]
        :param cur_state: 튜플 (h_cur, c_cur) - 이전 시점의 은닉 상태와 셀 상태
        """
        h_cur, c_cur = cur_state

        # 1. 입력(Xt)과 이전 은닉 상태(Ht-1)를 채널 차원(dim=1)으로 결합 (Concatenate)
        # 논문의 수식에서 W_xi * X + W_hi * H 부분을 효율적으로 처리하는 방식입니다.
        combined = torch.cat([input_tensor, h_cur], dim=1)

        # 2. 합성곱 연산 수행
        combined_conv = self.conv(combined)

        # 3. 결과 텐서를 4개의 게이트로 분할 (cc_i, cc_f, cc_o, cc_g)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        # 4. 활성화 함수 적용 (논문의 수식 (3) 참조 )
        i = torch.sigmoid(cc_i) # Input Gate
        f = torch.sigmoid(cc_f) # Forget Gate
        o = torch.sigmoid(cc_o) # Output Gate
        g = torch.tanh(cc_g)    # Cell Input (Input modulation)

        # 5. 셀 상태 업데이트 (C_t)
        # 논문 수식: C_t = f_t * C_{t-1} + i_t * tanh(...)
        # * 연산은 요소별 곱(Hadamard Product)입니다.
        c_next = f * c_cur + i * g

        # 6. 은닉 상태 업데이트 (H_t)
        # 논문 수식: H_t = o_t * tanh(C_t)
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        """
        초기 상태(0) 생성 함수
        논문에서는 초기 상태를 0으로 설정하는 것이 '미래에 대한 완전한 무지'를 의미한다고 설명합니다[cite: 109].
        """
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class SensorConvLSTM(nn.Module):
    def __init__(self, hidden_dim=64, kernel_size=3, num_classes=10):
        """
        :param hidden_dim: ConvLSTM 내부의 은닉 상태 채널 수
        :param kernel_size: 커널 크기 (홀수 권장, 예: 3은 좌우 센서 1개씩을 함께 봄)
        :param num_classes: 분류할 클래스 개수
        """
        super(SensorConvLSTM, self).__init__()
        
        self.num_sensors = 21  # 센서 개수 (Spatial Width)
        self.input_channels = 3 # x, y, z (Input Depth)
        self.hidden_dim = hidden_dim
        
        # 1. ConvLSTM Cell 초기화
        # 높이가 1이므로 커널의 높이도 1로 고정 (1, kernel_size)
        # 패딩을 적절히 주어 센서 개수(21)가 유지되도록 함
        padding = (0, kernel_size // 2) 
        self.conv_lstm = ConvLSTMCell(input_dim=self.input_channels, 
                                      hidden_dim=hidden_dim, 
                                      kernel_size=(1, kernel_size))
        
        # 2. 분류기 (Classifier)
        # Global Average Pooling 후 입력되므로 input_dim = hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        :param x: (Batch, Time, 63) - Flatten된 입력
        """
        b, t, d = x.size()
        
        # [Step 1] 데이터 Reshape: (Batch, Time, 63) -> (Batch, Time, 21, 3)
        # 63개의 값을 21개 센서의 3개 좌표로 분리
        x = x.view(b, t, self.num_sensors, self.input_channels)
        
        # [Step 2] Permute: (Batch, Time, 21, 3) -> (Batch, Time, 3, 1, 21)
        # ConvLSTM은 (B, T, Channel, Height, Width)를 원하므로
        # Channel=3 (x,y,z), Height=1, Width=21로 맞춤
        x = x.permute(0, 1, 3, 2).unsqueeze(3)
        # x shape: (Batch, Time, 3, 1, 21)
        
        # 초기 상태 초기화 (H=1, W=21)
        h, c = self.conv_lstm.init_hidden(b, (1, self.num_sensors))
        
        # [Step 3] 시퀀스 처리 (Time loop)
        for step in range(t):
            # 현재 프레임: (Batch, 3, 1, 21)
            current_input = x[:, step, :, :, :]
            h, c = self.conv_lstm(current_input, (h, c))
        
        # 루프 종료 후 h의 shape: (Batch, Hidden_Dim, 1, 21)
        # 이 h는 "각 센서 위치별로 압축된 시간적 특징"을 담고 있음
        
        # [Step 4] Feature Aggregation
        # 모든 센서(21개)의 정보를 평균내어 하나의 벡터로 만듦 (Global Average Pooling)
        # dim=3(Width=21) 방향으로 평균
        feature_vector = torch.mean(h, dim=3).squeeze(2) # (Batch, Hidden_Dim)
        
        # [Step 5] 최종 분류
        output = self.classifier(feature_vector)
        
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

def run_kfold_experiment(dataset, stroke_type, joint_type, body_part, k=5, epoch = 20, batch_size=16,
                         hidden_dim=32, kernel_size=3, num_classes=5, device='cuda', output_dir='./results'):

    os.makedirs(output_dir, exist_ok=True)
    
    tsne_save_dir = os.path.join(output_dir, f'tsne_images_{batch_size}_{hidden_dim}_{kernel_size}_{num_classes}')
    os.makedirs(tsne_save_dir, exist_ok=True)

    folds, labels = dataset.split_data_Kfold_randomly(stroke_type, k, body_part)
    accumulated_accuracy = []

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

        model = SensorConvLSTM(hidden_dim=hidden_dim, kernel_size=kernel_size, num_classes=num_classes).to(device)
        
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

        with torch.no_grad(): # 그래디언트 계산 비활성화 (메모리 절약, 속도 향상)
            for inputs, test_labels in test_loader:
                inputs, test_labels = inputs.to(device), test_labels.to(device)
                
                outputs = model(inputs) # 모델 예측값 (Logits)
                
                # 가장 높은 확률을 가진 클래스 인덱스 추출
                # _: max value (확률값), predicted: max index (예측 클래스 번호)
                _, predicted = torch.max(outputs.data, 1)
                
                total += test_labels.size(0) # 전체 샘플 수
                correct += (predicted == test_labels).sum().item() # 맞은 개수 누적
        accuracy = correct / total
        accumulated_accuracy.append(accuracy)
    
    average_accuracy = sum(accumulated_accuracy) / k
    
    print(f"\n{'='*30}")
    print(f"Final Result ({k}-Fold CV)")
    print(f"Avg Accuracy: {average_accuracy:.4f}")
    print(f"{'='*30}")

    result_summary = {
        'experiment': f"{stroke_type}_{joint_type}_{body_part}_kfold",
        'parameter': f"{batch_size}, {hidden_dim}, {kernel_size}, {num_classes}",
        'fold_accuracies': average_accuracy, # 스칼라 값 저장
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
    output_dir = './results_convlstm'
    
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
                epoch=1,batch_size=64, hidden_dim=32, kernel_size=3, num_classes=7)
            
        except Exception as e:
            print(f"\nERROR in experiment {stroke_type}_{joint_type}_{body_part}: {e}")
            import traceback
            traceback.print_exc()
        
    print(f"\nAll results saved to {output_dir}")


if __name__ == '__main__':
    main()