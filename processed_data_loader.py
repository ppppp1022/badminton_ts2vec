import numpy as np
import os
from typing import List, Tuple
import random
import torch

from preprocess_badminton_data import load_skill_levels_from_annotation

class ProcessedBadmintonDataset:
    """전처리된 배드민턴 데이터셋 로더 (너의 파일구조에 맞게 수정됨)"""
    
    def __init__(self, data_folder: str):
        self.data_folder = data_folder
        
        # Skill level 리스트
        self.clear_skill_level = []
        self.drive_skill_level = []
        
        # Skill level 그룹
        self.beginner_subjects = []
        self.intermediate_subjects = []
        self.expert_subjects = []
        
        # Joint index 설정
        self.local_arm_index = []
        self.global_arm_index = []
        self.local_leg_index = []
        self.global_leg_index = []
        self.local_total_index = []
        self.global_total_index = []
    
    def load_subject_data(self, subject_id: str, stroke_type: str, 
                          joint_type: str, body_part: str) -> Tuple[List[np.ndarray], float]:
        """
        너의 디렉토리 구조:
        """
        
        # 예: S00_clear
        folder_name = f"{subject_id}_{stroke_type}"
        subject_dir = os.path.join(self.data_folder, subject_id, folder_name)
        
        if joint_type == 'local':
            file_path = os.path.join(subject_dir, "local.npy")
        else:
            file_path = os.path.join(subject_dir, "global.npy")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[ERROR] File not found: {file_path}")
        
        # npy file에는 list of strokes가 저장됨 (shape: (num_strokes, 150, 63))
        stroke_data = np.load(file_path, allow_pickle=True)

        # ----------------------------------
        # 2) body part → joint index 선택
        # ----------------------------------
        if body_part == 'arm':
            joint_indices = self.local_arm_index if joint_type == 'local' else self.global_arm_index
        elif body_part == 'leg':
            joint_indices = self.local_leg_index if joint_type == 'local' else self.global_leg_index
        else:  # total
            joint_indices = self.local_total_index if joint_type == 'local' else self.global_total_index

        # ----------------------------------
        # 3) stroke별 joint 추출
        # ----------------------------------
        processed_strokes = []

        for stroke in stroke_data:
            # (150, 63) → (150, 21, 3)
            stroke_reshaped = stroke.reshape(-1, 21, 3)

            # joint subset 선택
            selected_joints = stroke_reshaped[:, joint_indices, :]

            # (150, num_joints, 3) → (150, num_joints * 3)
            selected_joints_flat = selected_joints.reshape(len(selected_joints), -1)

            processed_strokes.append(selected_joints_flat)
        
        # ----------------------------------
        # 4) skill level 선택
        # ----------------------------------
        subject_idx = int(subject_id[1:])   # S00 → 0
        
        if stroke_type == 'clear':
            skill = self.clear_skill_level[subject_idx]
        else:
            skill = self.drive_skill_level[subject_idx]
        
        return processed_strokes, skill
    
    def split_data_by_skill(self, stroke_type: str):
        """스킬 그룹별로 train/test subject 분리"""
        
        test_b = random.choice(self.beginner_subjects)
        test_m = random.choice(self.intermediate_subjects)
        test_e = random.choice(self.expert_subjects)
        
        test_subjects = [test_b, test_m, test_e]
        
        all_subjects = self.beginner_subjects + self.intermediate_subjects + self.expert_subjects
        train_subjects = [s for s in all_subjects if s not in test_subjects]
        
        skill_list = self.clear_skill_level if stroke_type == 'clear' else self.drive_skill_level
        
        train_labels = [skill_list[int(s[1:])] for s in train_subjects]
        test_labels = [skill_list[int(s[1:])] for s in test_subjects]
        
        return train_subjects, test_subjects, train_labels, test_labels

    def split_data_Kfold(self, stroke_type: str, k: int = 5):
        """K-fold, 랜덤하게 subject 고른 후 분리"""
        subjects = [f"S{i:02d}" for i in range(25)]
        random.shuffle(subjects)
        fold_size = len(subjects) // k

        skill_list = self.clear_skill_level if stroke_type == 'clear' else self.drive_skill_level

        folds = [subjects[i*fold_size:(i+1)*fold_size] for i in range(k)]
        labels = []
        for fold in folds:
            fold_labels = [skill_list[int(s[1:])] for s in fold]
            labels.append(fold_labels)
        return folds, labels
    
    def split_data_Kfold_randomly(self, stroke_type: str, joint_type: str, body_part: List, k: int = 5):
        """K-fold, 통합 후 랜덤하게 분리"""
        folds = []
        all_strokes = []
        labels = []
        subjects = self.beginner_subjects + self.intermediate_subjects + self.expert_subjects
        
        for subj in subjects:
            strokes, skills = self.load_subject_data(subj, stroke_type, joint_type, body_part)
            all_strokes.extend(strokes)
            labels.extend([skills] * len(strokes))
        
        indices = torch.randperm(len(all_strokes)).tolist()
        all_strokes = [all_strokes[i] for i in indices]
        labels = [labels[i] for i in indices]

        n_samples = len(all_strokes)
        folds_strokes = []
        folds_labels = []

        for i in range(k):
            start_idx = (n_samples * i) // k
            end_idx = (n_samples * (i + 1)) // k
            # 데이터 슬라이싱하여 저장
            # all_strokes[start:end] -> [stroke_A, stroke_B, ...] 형태
            folds_strokes.append(all_strokes[start_idx:end_idx])
            folds_labels.append(labels[start_idx:end_idx])

        return folds_strokes, folds_labels

    def get_statistics(self):
        """데이터셋 통계 출력 (너의 파일 형식에 맞게 수정됨)"""
        print("\n=== Dataset Statistics ===")
        
        for i in range(25):
            subject_id = f"S{i:02d}"
            subject_dir = os.path.join(self.data_folder, subject_id)

            if not os.path.exists(subject_dir):
                continue

            clear_dir = os.path.join(subject_dir, f"{subject_id}_clear")
            drive_dir = os.path.join(subject_dir, f"{subject_id}_drive")

            clear_count = 0
            drive_count = 0

            if os.path.exists(os.path.join(clear_dir, "local.npy")):
                clear_data = np.load(os.path.join(clear_dir, "local.npy"), allow_pickle=True)
                clear_count = len(clear_data)

            if os.path.exists(os.path.join(drive_dir, "local.npy")):
                drive_data = np.load(os.path.join(drive_dir, "local.npy"), allow_pickle=True)
                drive_count = len(drive_data)

            if clear_count > 0 or drive_count > 0:
                clear_skill = self.clear_skill_level[i] if i < len(self.clear_skill_level) else -1
                drive_skill = self.drive_skill_level[i] if i < len(self.drive_skill_level) else -1

                print(f"{subject_id}: Clear={clear_count} (skill={clear_skill:.2f}), "
                      f"Drive={drive_count} (skill={drive_skill:.2f})")


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

# 사용 예시
if __name__ == '__main__':
    # 데이터셋 초기화
    processed_data_folder = './Processed_Data'
    dataset = setup_dataset(processed_data_folder)
    dataset.get_statistics()
    