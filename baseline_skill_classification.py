from Networks.SciDataModels import Conv1DRefinedModel
import torch
import torch.nn as nn
import numpy as np
from processed_data_loader import ProcessedBadmintonDataset
from preprocess_badminton_data import load_skill_levels_from_annotation

def train(model, X, y, epochs=50, batch_size=64):
    model.train()
    N = X.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(N)
        X = X[perm]
        y = y[perm]
        
        total_loss = 0
        
        for i in range(0, N, batch_size):
            xb = X[i:i+batch_size]
            yb = y[i:i+batch_size]
            print(f"xb: {xb.shape}, yb: {yb.shape}")
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}  Loss: {total_loss:.4f}")

def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(dim=1)
        acc = (pred == y).float().mean().item()
    
    print("Accuracy:", acc)
    return pred.cpu().numpy()




# ---------------------------
# Init parameters
# ---------------------------
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# input_feature_matrices: (N, L, D) 형태라고 가정
# hidden_features, label_num은 이미 정의되어 있다고 가정

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

for stroke_type, joint_type, body_part in experiments:
    train_subjects, test_subjects, train_labels, test_labels = dataset.split_data_by_skill(stroke_type)
    
    train_data_list = []
    for subject in train_subjects:
        data, _ = dataset.load_subject_data(subject, stroke_type, joint_type, body_part)
        train_data_list.append(data)

    # test data load
    test_data_list = []
    for subject in test_subjects:
        data, _ = dataset.load_subject_data(subject, stroke_type, joint_type, body_part)
        test_data_list.append(data)

        sample_data = train_data_list[0][0]
        input_dim = sample_data.shape[1]
        hidden_dim = 64                  # 사용자가 설정한 hidden dim
        output_dim = 7                        # 분류 클래스 수

        # ---------------------------
        # Create model
        # ---------------------------
        model = Conv1DRefinedModel(
            input_dim,
            hidden_dim,
            output_size=output_dim
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        