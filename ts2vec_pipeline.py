"""
Complete pipeline for badminton skill level classification using TS2Vec

Steps:
1. Load annotation file and extract skill levels
2. Load preprocessed data
3. Train TS2Vec for representation learning
4. Train skill level regressor
5. Evaluate on test subjects
"""

import os
import json
import numpy as np
import torch
from preprocess_badminton_data import load_skill_levels_from_annotation
from processed_data_loader import ProcessedBadmintonDataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from ts2vec_skill_classification import (
    TS2Vec, TS2VecTrainer, SkillLevelClassifier, SkillLevelTrainer
)
# TS2Vec 관련 import는 기존 artifact에서 가져온다고 가정

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

def augment_dataset(data_list, labels_list, num_augments=1):
    """
    데이터셋을 증강하여 크기를 키우는 함수
    Args:
        data_list: 원본 데이터 리스트
        labels_list: 원본 라벨 리스트
        num_augments: 원본 1개당 생성할 증강 데이터 개수 (예: 1이면 2배가 됨)
    """
    augmented_data = []
    augmented_labels = []
    
    for data, label in zip(data_list, labels_list):
        # 1. 원본은 무조건 포함
        augmented_data.append(data)
        augmented_labels.append(label)
        
        # 2. 증강 데이터 생성
        for _ in range(num_augments):
            # (1) Jittering (노이즈)
            noise = np.random.normal(0, 0.02, data[0].shape)
            new_data = data + noise
            
            # (2) Scaling (크기 변형)
            scale = np.random.normal(1.0, 0.1) # 0.9 ~ 1.1 배
            new_data = new_data * scale
            
            # (3) 추가적인 변형이 있다면 여기서 적용 (Velocity 변환 전이라면 효과 좋음)
            
            augmented_data.append(new_data)
            augmented_labels.append(label)
            
    return augmented_data, augmented_labels

def run_experiment(dataset, stroke_type, joint_type, body_part, device='cuda', 
                   ts2vec_epochs=100, classifier_epochs=50, output_dir='./results'):
    """
    하나의 실험 실행 (특정 stroke_type, joint_type, body_part 조합)
    
    Args:
        dataset: ProcessedBadmintonDataset 객체
        stroke_type: 'clear' or 'drive'
        joint_type: 'local' or 'global'
        body_part: 'arm', 'leg', 'total'
        device: 'cuda' or 'cpu'
        ts2vec_epochs: TS2Vec 학습 에폭 수
        classifier_epochs: Classifier 학습 에폭 수
        output_dir: 결과 저장 폴더
    """
    from ts2vec_skill_classification import (
        TS2Vec, TS2VecTrainer, SkillLevelClassifier, SkillLevelTrainer
    )
    
    print(f"Experiment: {stroke_type} - {joint_type} - {body_part}")
    
    # 1. 데이터 분할
    train_subjects, test_subjects, train_labels, test_labels = dataset.split_data_by_skill(stroke_type)
    
    print(f"\nTrain subjects ({len(train_subjects)}): {train_subjects}")
    print(f"Train labels (float): {[f'{l:.2f}' for l in train_labels]}")
    print(f"\nTest subjects ({len(test_subjects)}): {test_subjects}")
    print(f"Test labels (float): {[f'{l:.2f}' for l in test_labels]}")
    
    # 2. 데이터 로드
    print("\nLoading training data...")
    train_data_list = []
    for subject in train_subjects:
        try:
            strokes, skill = dataset.load_subject_data(subject, stroke_type, joint_type, body_part)
            train_data_list.append(strokes)
            #print(f"  {subject}: {len(strokes)} strokes, skill={skill:.2f}")
        except Exception as e:
            print(f"  {subject}: Error - {e}")
    
    print("\nLoading test data...")
    test_data_list = []
    for subject in test_subjects:
        try:
            strokes, skill = dataset.load_subject_data(subject, stroke_type, joint_type, body_part)
            test_data_list.append(strokes)
            #print(f"  {subject}: {len(strokes)} strokes, skill={skill:.2f}")
        except Exception as e:
            print(f"  {subject}: Error - {e}")
    
    if len(train_data_list) == 0 or len(test_data_list) == 0:
        print("ERROR: No data loaded. Skipping this experiment.")
        return None
    
    # 3. Input dimension 계산
    sample_data = train_data_list[0][0]
    input_dim = sample_data.shape[1]
    print(f"\nInput dimension: {input_dim}")
    
    # 4. TS2Vec 학습
    print("\n" + "="*60)
    print("Training TS2Vec")
    print("="*60)
    
    ts2vec_model = TS2Vec(input_dim=input_dim, hidden_dim= 256, output_dim=256)
    ts2vec_trainer = TS2VecTrainer(ts2vec_model, lr=0.001, device=device)
    #print(f"train data list shape: {type(train_data_list)}\n")
    for epoch in range(ts2vec_epochs):
        loss = ts2vec_trainer.train_epoch(train_data_list, batch_size=32)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{ts2vec_epochs}, Loss: {loss:.4f}')
    
    print("\n Saving TS2Vec Model")
    model_path = ts2vec_trainer.save_model(stroke_type, joint_type)
    print(f"TS2Vec model saved to {model_path}")
    # 5. 임베딩 추출
    print("Extracting Embeddings")
    
    skill_trainer = SkillLevelTrainer(
        ts2vec_model,
        SkillLevelClassifier(embedding_dim=128),
        lr=0.001,
        device=device
    )
    
    train_embeddings = skill_trainer.extract_embeddings(train_data_list)
    test_embeddings = skill_trainer.extract_embeddings(test_data_list)
    
    # Label 확장
    train_labels_expanded = []
    for i, data in enumerate(train_data_list):
        train_labels_expanded.extend([train_labels[i]] * len(data))
    train_labels_expanded = np.array(train_labels_expanded, dtype=np.float32)
    
    test_labels_expanded = []
    for i, data in enumerate(test_data_list):
        test_labels_expanded.extend([test_labels[i]] * len(data))
    test_labels_expanded = np.array(test_labels_expanded, dtype=np.float32)
    
    print(f"Train embeddings: {train_embeddings.shape}")
    print(f"Test embeddings: {test_embeddings.shape}")
    print(f"Train labels range: {train_labels_expanded.min():.2f} - {train_labels_expanded.max():.2f}")
    
    # 6. Skill Level Regressor 학습
    print("Training Skill Level Regressor")
    
    for epoch in range(classifier_epochs):
        loss = skill_trainer.train_epoch(train_embeddings, train_labels_expanded, batch_size=64)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{classifier_epochs}, MSE Loss: {loss:.4f}')
    
    # 7. 평가
    print("Evaluation")
    
    results = skill_trainer.evaluate(test_embeddings, test_labels_expanded)
    
    print(f"Accuracy (rounded): {results['accuracy']:.4f}")
    print(f"\npredictions: {results['predictions']}")
    print(f"rmse:{results['rmse']:.4f}")
    print(results['report'])
    print("\nConfusion Matrix (rounded predictions):")
    print(results['confusion_matrix'])
    
    # 8. 피험자별 평가
    print("Per-Subject Evaluation")
    
    subject_results = []
    start_idx = 0
    for i, (subject, data) in enumerate(zip(test_subjects, test_data_list)):
        end_idx = start_idx + len(data)
        subject_preds = results['predictions'][start_idx:end_idx]
        subject_true = test_labels[i]
        
        avg_pred = subject_preds.mean()
        rounded_pred = round(avg_pred)
        error = abs(avg_pred - subject_true)
        
        subject_result = {
            'subject': subject,
            'true': subject_true,
            'predicted': avg_pred,
            'rounded': rounded_pred,
            'error': error
        }
        subject_results.append(subject_result)
        
        print(f"{subject}: True={subject_true:.2f}, Predicted={avg_pred:.2f} "
              f"(rounded: {rounded_pred}), Error={error:.2f}")
        
        start_idx = end_idx
    
    # 9. 결과 저장
    os.makedirs(output_dir, exist_ok=True)
    
    result_summary = {
        'experiment': f"{stroke_type}_{joint_type}_{body_part}",
        'train_subjects': train_subjects,
        'test_subjects': test_subjects,
        'metrics': {
            'prediction': results['predictions'].tolist(),
            'accuracy': float(results['accuracy']),
            'rmse': float(results['rmse'])
        },
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'subject_results': subject_results
    }
    
    result_filename = f"{stroke_type}_{joint_type}_{body_part}_results.json"
    with open(os.path.join(output_dir, result_filename), 'w') as f:
        json.dump(result_summary, f, indent=2)
    
    print(f"\nResults saved to {os.path.join(output_dir, result_filename)}")
    
    return result_summary

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
    # NaN 값 안전 장치 (이전 에러 방지)
    if np.isnan(embeddings).any():
        print(f"⚠️ Warning: NaN found in {suffix} embeddings. Replacing with 0 for visualization.")
        embeddings = np.nan_to_num(embeddings)

    print(f"Generating T-SNE for Fold {fold_idx} ({suffix})...")
    
    # T-SNE 설정 (init='random'으로 안정성 확보)
    tsne = TSNE(
        n_components=2, 
        random_state=42, 
        perplexity=30, 
        max_iter=1000,
        init='random',       # [중요] NaN 에러 방지용
        learning_rate='auto'
    )
    
    # 데이터가 너무 많으면 시간이 오래 걸리므로 3000개 정도로 샘플링하는 것도 방법
    # 여기선 일단 전체 다 그립니다.
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # 시각화 데이터프레임
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
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

def run_kfold_experiment(dataset, stroke_type, joint_type, body_part, k=5, device='cuda', 
                          ts2vec_epochs=100, classifier_epochs=50, output_dir='./results'):
    """
    K-Fold 교차 검증 실험 실행 (T-SNE 포함)
    """
    def normalize_data(data_list, method='standard'):
        """
        데이터 정규화 함수 (Step 2)
        Args:
            data_list: List of np.array (각 array는 [Time, Features])
            method: 'standard' (평균0, 표준편차1) or 'minmax' (0~1)
        Returns:
            normalized_list: 정규화된 데이터 리스트
        """
        normalized_list = []
        
        for stroke in data_list:
            # stroke shape: (Time, Features)
            
            if method == 'standard':
                # [추천] Standard Scaling (Z-score)
                # 각 Feature별로 시간축(Time)에 대한 평균/표준편차 계산
                mean = np.mean(stroke, axis=0)
                std = np.std(stroke, axis=0)
                
                # 분모 0 방지 (매우 중요!)
                std = np.where(std == 0, 1e-6, std)
                
                norm_stroke = (stroke - mean) / std
                
            elif method == 'minmax':
                # Min-Max Scaling (-1 ~ 1 또는 0 ~ 1)
                min_val = np.min(stroke, axis=0)
                max_val = np.max(stroke, axis=0)
                
                denom = max_val - min_val
                denom = np.where(denom == 0, 1e-6, denom)
                
                # -1 ~ 1 로 변환 (Global 좌표에 적합)
                norm_stroke = 2 * (stroke - min_val) / denom - 1
                
            normalized_list.append(norm_stroke)
            
        return normalized_list
    # 필요한 클래스들이 이미 import 되어 있다고 가정합니다.
    # from models import TS2Vec, TS2VecTrainer, SkillLevelClassifier, SkillLevelTrainer
    
    # 결과 저장용 폴더 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. [추가] 이미지 저장용 하위 폴더 생성
    tsne_save_dir = os.path.join(output_dir, 'tsne_images')
    os.makedirs(tsne_save_dir, exist_ok=True)

    folds, labels = dataset.split_data_Kfold(stroke_type, k)
    accumulated_accuracy = []
    mean_errors = []
    
    print(f"Starting experiment: {stroke_type}_{joint_type}_{body_part}")

    for fold_idx in range(k):
        print(f"\n{'='*20}\n Fold {fold_idx+1} / {k} \n{'='*20}")
        
        # --- 1. 데이터 준비 ---
        train_subjects = [s for i, fold in enumerate(folds) if i != fold_idx for s in fold]
        test_subjects = folds[fold_idx]
        train_labels = [l for i, label in enumerate(labels) if i != fold_idx for l in label]
        test_labels = labels[fold_idx]

        train_data_list = []
        for subject in train_subjects:
            strokes, skill = dataset.load_subject_data(subject, stroke_type, joint_type, body_part)
            train_data_list.append(strokes)
        
        test_data_list = []
        for subject in test_subjects:
            strokes, skill = dataset.load_subject_data(subject, stroke_type, joint_type, body_part)
            test_data_list.append(strokes)
        

        print(f"===Data Augumentation===")
        train_data_list, train_labels = augment_dataset(
            train_data_list, 
            train_labels, 
            num_augments=2 
        )

        print(f"{len(train_data_list)}")
        print(f"===Normalization===")
        train_data_list = normalize_data(train_data_list, method='standard')
        test_data_list = normalize_data(test_data_list, method='standard')

        # Input Dimension 확인
        sample_data = train_data_list[0]
        input_dim = sample_data.shape[2]  
        print(f"Input dimension: {input_dim}")
        # --- 2. TS2Vec 학습 (Unsupervised) ---
        print("\n[Step 1] Training TS2Vec (Self-Supervised)...")
        
        # hidden_dim=128로 설정 (Classifier 입력과 맞춤)
        ts2vec_model = TS2Vec(input_dim=input_dim, hidden_dim=64, output_dim=256)
        ts2vec_trainer = TS2VecTrainer(ts2vec_model, lr=0.001, device=device)
        
        for epoch in range(ts2vec_epochs):
            loss = ts2vec_trainer.train_epoch(train_data_list, batch_size=128)
            if (epoch + 1) % 10 == 0:
                print(f'  Epoch {epoch+1}/{ts2vec_epochs}, Loss: {loss:.4f}')
        
        # --- 3. 임베딩 추출 및 분류기 학습 ---
        print("\n[Step 2] Extracting Embeddings & Training Classifier...")
        
        # output_dim=128은 TS2Vec의 hidden_dim과 일치해야 함
        skill_trainer = SkillLevelTrainer(
            ts2vec_model,
            SkillLevelClassifier(embedding_dim=64, num_classes=7), # num_classes 명시 권장
            lr=0.001,
            device=device
        )
        
        # 임베딩 추출 (배치 처리된 개선 버전 사용 권장)
        train_embeddings = skill_trainer.extract_embeddings(train_data_list)
        test_embeddings = skill_trainer.extract_embeddings(test_data_list)

        # 라벨 확장 (Subject 단위 -> Stroke 단위)
        train_labels_expanded = []
        for i, data in enumerate(train_data_list):
            train_labels_expanded.extend([train_labels[i]] * len(data))
        train_labels_expanded = np.array(train_labels_expanded, dtype=np.float32) # or int
        
        test_labels_expanded = []
        for i, data in enumerate(test_data_list):
            test_labels_expanded.extend([test_labels[i]] * len(data))
        test_labels_expanded = np.array(test_labels_expanded, dtype=np.float32)
        print("\n[Visualization] Generating T-SNE plots...")
        
        # 1. Train Data 시각화
        visualize_embeddings(
            train_embeddings, 
            train_labels_expanded, 
            fold_idx=fold_idx+1, 
            save_dir=tsne_save_dir, # 하위 폴더 지정
            suffix="Train"
        )
        
        # 2. Test Data 시각화
        visualize_embeddings(
            test_embeddings, 
            test_labels_expanded, 
            fold_idx=fold_idx+1, 
            save_dir=tsne_save_dir, # 하위 폴더 지정
            suffix="Test"
        )

        # 분류기 학습
        for epoch in range(classifier_epochs):
            loss = skill_trainer.train_epoch(train_embeddings, train_labels_expanded, batch_size=64)
            if (epoch + 1) % 10 == 0:
                print(f'  Epoch {epoch+1}/{classifier_epochs}, Classifier Loss: {loss:.4f}')

        # --- 4. 평가 (Evaluation) ---
        print("\n[Step 3] Evaluating...")
        results = skill_trainer.evaluate(test_embeddings, test_labels_expanded)
        print(f"Fold {fold_idx+1} Accuracy: {results['accuracy']:.4f}")
        
        # --- 5. Subject 단위 결과 집계 ---
        print("\n[Per-Subject Analysis]")
        subject_results = []
        start_idx = 0
        
        for i, (subject, data) in enumerate(zip(test_subjects, test_data_list)):
            end_idx = start_idx + len(data)
            
            # 해당 Subject의 예측값들 슬라이싱
            subject_preds = results['predictions'][start_idx:end_idx]
            subject_true = test_labels[i]
            
            # 평균 예측값 계산
            avg_pred = subject_preds.mean()
            rounded_pred = round(avg_pred)
            error = abs(avg_pred - subject_true)
            
            subject_results.append({
                'subject': subject,
                'true': subject_true,
                'predicted': avg_pred,
                'rounded': rounded_pred,
                'error': error
            })
            
            print(f"  Subject {subject}: True={subject_true}, Pred={avg_pred:.2f} "
                  f"(Round: {rounded_pred}), Error={error:.2f}")
            
            start_idx = end_idx

        # Fold별 평균 에러 저장
        fold_mean_error = np.mean([res['error'] for res in subject_results])
        mean_errors.append(fold_mean_error)
        accumulated_accuracy.append(results['accuracy'])

    # --- 6. 전체 결과 요약 및 저장 ---
    final_mean_error = np.mean(mean_errors)
    average_accuracy = sum(accumulated_accuracy) / k
    
    print(f"\n{'='*30}")
    print(f"Final Result ({k}-Fold CV)")
    print(f"Avg Accuracy: {average_accuracy:.4f}")
    print(f"Avg Error: {final_mean_error:.4f}")
    print(f"{'='*30}")

    result_summary = {
        'experiment': f"{stroke_type}_{joint_type}_{body_part}_kfold",
        'fold_accuracies': average_accuracy, # 스칼라 값 저장
        'fold_accuracies_list': accumulated_accuracy, # 상세 기록용 리스트도 저장 추천
        'error': final_mean_error
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
    """메인 실행 함수"""
    # 경로 설정
    processed_data_folder = './Processed_Data'
    output_dir = './results_kfold'
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 데이터셋 설정
    dataset = setup_dataset(processed_data_folder)
    
    # 실험 조합 정의
    experiments = [
        # Clear - 전체
        #('clear', 'global', 'total'),
        #('clear', 'local', 'total'),
        
        # Clear - 부위별
        ('clear', 'global', 'arm'),
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
                ts2vec_epochs=150,
                classifier_epochs=100,
                output_dir=output_dir
            )
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"\nERROR in experiment {stroke_type}_{joint_type}_{body_part}: {e}")
            import traceback
            traceback.print_exc()
        
    # 전체 결과 저장
    
    print(f"\nAll results saved to {output_dir}")

if __name__ == '__main__':
    main()