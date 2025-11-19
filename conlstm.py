import torch
import torch.nn as nn
import numpy as np
import random
from preprocess_badminton_data import load_skill_levels_from_annotation
from processed_data_loader import ProcessedBadmintonDataset
import os

class ConvLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, lstm_hidden=128, num_classes=7):
        super().__init__()
        self.input_dim = input_dim
        # 1D CNN feature extractor
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        features = self.conv(x)  # (batch, hidden_dim, seq_len)
        features = features.transpose(1, 2)  # (batch, seq_len, hidden_dim)
        
        lstm_out, _ = self.lstm(features)
        pooled = lstm_out.mean(dim=1)  # temporal average pooling
        
        output = self.fc(pooled)
        return output
    def save_model(self, model, stroke_type: str, joint_type: str, save_dir='./CLSTMModel'):
        """
        CovLSTM 모델 저장
        
        Args:
            stroke_type: 'clear' or 'drive'
            joint_type: 'local' or 'global'
            save_dir: 저장 디렉토리
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 파일명: ts2vec_clear_global.pth
        model_name = f"clstm_{stroke_type}_{joint_type}.pth"
        model_path = os.path.join(save_dir, model_name)
        
        # 모델 state와 설정 저장
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_dim': model.input_dim,
            },
            'stroke_type': stroke_type,
            'joint_type': joint_type
        }
        
        torch.save(checkpoint, model_path)
        print(f"Model saved to {model_path}")
        
        return model_path
    
    @staticmethod
    def load_model(stroke_type: str, joint_type: str, device='cuda', save_dir='./CLSTMModel'):
        """
        저장된 clstm 모델 로드
        
        Args:
            stroke_type: 'clear' or 'drive'
            joint_type: 'local' or 'global'
            device: 'cuda' or 'cpu'
            save_dir: 저장 디렉토리
            
        Returns:
            model: 로드된 TS2Vec 모델
            trainer: TS2VecTrainer 객체
        """
        model_name = f"ts2vec_{stroke_type}_{joint_type}.pth"
        model_path = os.path.join(save_dir, model_name)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Checkpoint 로드
        checkpoint = torch.load(model_path, map_location=device)
        
        # 모델 재생성
        config = checkpoint['model_config']
        model = ConvLSTMClassifier(
            input_dim=config['input_dim'],
            num_classes=7
        )
        
        # State 로드
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        return model

def train_clstm(model, train_data_list, train_labels, device='mps', epochs=50, batch_size=64):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    strokes = []
    labels = []
    for subj_data, label in zip(train_data_list, train_labels):
        for stroke in subj_data:
            strokes.append(stroke)
            labels.append(label - 1)

    X = np.array(strokes, dtype=np.float32)   # (num_strokes, seq_len, input_dim)
    X = torch.from_numpy(X).to(device)
    
    y = np.array(labels, dtype=np.int64)
    y = torch.from_numpy(y).to(device)

    for epoch in range(epochs):
        perm = torch.randperm(len(X))
        X = X[perm]
        y = y[perm]

        total_loss = 0
        for i in range(0, len(X), batch_size):
            xb = X[i:i+batch_size]
            yb = y[i:i+batch_size]

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}')
def evaluate_clstm(model, test_data_list, test_labels, device='cuda'):
    model.eval()
    
    strokes = []
    labels = []
    for subj_data, label in zip(test_data_list, test_labels):
        for stroke in subj_data:
            strokes.append(stroke)
            labels.append(label)

    # Convert labels to int (very important)
    labels = np.array(labels, dtype=int)

    X = torch.FloatTensor(strokes).to(device)

    with torch.no_grad():
        preds = model(X).argmax(dim=1).cpu().numpy() + 1

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    print(f"Test Results: {preds}\nTrue Labels: {labels}")

    acc = accuracy_score(labels, preds)
    print("Accuracy:", acc)
    print(classification_report(labels, preds))
    print(confusion_matrix(labels, preds))

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_experiment(dataset, stroke_type, joint_type, body_part,
                   device='cuda', epochs=100, batch_size=64,
                   output_dir='./results'):
    """
    전체 실험을 실행하는 고수준 함수
    - stroke_type: 'clear' or 'drive'
    - joint_type: 'local' or 'global'
    - body_part: 'arm', 'leg', 'total'
    """

    # ---------------------------
    # 1) 폴더 생성
    # ---------------------------
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n===== Run Experiment =====")
    print(f"Stroke Type: {stroke_type}")
    print(f"Joint Type : {joint_type}")
    print(f"Body Part  : {body_part}")
    print(f"Device     : {device}")
    print("==========================\n")

    # ---------------------------
    # 2) 데이터 split by skill
    # ---------------------------
    train_subjects, test_subjects, train_labels, test_labels = dataset.split_data_by_skill(stroke_type)

    # train data load
    train_data_list = []
    for subject in train_subjects:
        data, _ = dataset.load_subject_data(subject, stroke_type, joint_type, body_part)
        train_data_list.append(data)

    # test data load
    test_data_list = []
    for subject in test_subjects:
        data, _ = dataset.load_subject_data(subject, stroke_type, joint_type, body_part)
        test_data_list.append(data)

    print(f"Train subjects loaded: {len(train_data_list)}")
    print(f"Test subjects loaded : {len(test_data_list)}")

    # ---------------------------
    # 3) 입력 차원 자동 추출
    # ---------------------------
    sample_data = train_data_list[0][0]
    input_dim = sample_data.shape[1]
    print(f"Input dimension = {input_dim}")

    model = ConvLSTMClassifier(input_dim=input_dim, num_classes=7)

    # ---------------------------
    # 4) Train
    # ---------------------------
    print("\n--- Training Start ---")
    train_clstm(
        model=model,
        train_data_list=train_data_list,
        train_labels=train_labels,
        device=device,
        epochs=epochs,
        batch_size=batch_size
    )

    # ---------------------------
    # 5) Evaluation
    # ---------------------------
    print("\n--- Evaluation ---")
    evaluate_clstm(
        model=model,
        test_data_list=test_data_list,
        test_labels=test_labels,
        device=device
    )

    # ---------------------------
    # 6) Save model
    # ---------------------------
    save_path = os.path.join(output_dir, f'CLSTM_{stroke_type}_{joint_type}_{body_part}.pt')
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to: {save_path}")


def main():
    set_seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print("Using device:", device)

    # ---------------------------
    # Load skill levels
    # ---------------------------
    clear_skills, drive_skills, subject_groups = load_skill_levels_from_annotation(
        annotation_filepath='./configs/skill_levels.json'
    )

    # ---------------------------
    # Load dataset
    # ---------------------------
    dataset = ProcessedBadmintonDataset('./Processed_Data')

    dataset.beginner_subjects = subject_groups['beginner']
    dataset.intermediate_subjects = subject_groups['intermediate']
    dataset.expert_subjects = subject_groups['expert']

    dataset.clear_skill_level = clear_skills
    dataset.drive_skill_level = drive_skills

    dataset.local_arm_index = [13, 14, 15, 16, 17, 18, 19, 20]  # Right/Left Shoulder, Arm, ForeArm, Hand
    dataset.global_arm_index = [13, 14, 15, 16, 17, 18, 19, 20]
    
    dataset.local_leg_index = [0, 1, 2, 3, 4, 5, 6]  # Hips + Legs
    dataset.global_leg_index = [0, 1, 2, 3, 4, 5, 6]
    
    dataset.local_total_index = list(range(21))  # All joints
    dataset.global_total_index = list(range(21))

    # ---------------------------
    # Run experiment
    # ---------------------------
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
    experiments = [
            # Clear - 전체
            ('clear', 'local', 'total')
        ]
    # 각 실험 실행
    all_results = []
    for stroke_type, joint_type, body_part in experiments:
        try:
            result = run_experiment(
                dataset,
                stroke_type,
                joint_type,
                body_part,
                device=device,
                epochs=200,
                batch_size=64,
                output_dir='./result_conlstm'
            )
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"\nERROR in experiment {stroke_type}_{joint_type}_{body_part}: {e}")
            import traceback
            traceback.print_exc()
        


if __name__ == "__main__":
    main()

