from ts2vec_skill_classification import (
        TS2Vec, TS2VecTrainer, SkillLevelClassifier, SkillLevelTrainer, AttentionSkillClassifier
    )
from main_pipeline import setup_dataset
import os
import json
import numpy as np
import torch
from preprocess_badminton_data import load_skill_levels_from_annotation
from processed_data_loader import ProcessedBadmintonDataset

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np
from sklearn.model_selection import KFold
import numpy as np

def run_kfold(embeddings, labels, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    fold = 1
    for train_idx, test_idx in kf.split(embeddings):
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]


        print(f"\n=== Fold {fold} ===")
        print("Train:", X_train.shape, " Test:", X_test.shape)

        

        fold += 1

def split_train_test(X, y, test_ratio=0.2, seed=42):
    """
    X : embedding vectors (N, D)
    y : labels (N,)
    test_ratio : 비율 (기본 0.2 → 8:2)
    """
    np.random.seed(seed)
    perm = np.random.permutation(len(X))
    
    X = X[perm]
    y = y[perm]
    
    test_size = int(len(X) * test_ratio)

    X_test = X[:test_size]
    y_test = y[:test_size]

    X_train = X[test_size:]
    y_train = y[test_size:]

    return X_train, X_test, y_train, y_test

def train_xgboost(X_train, y_train, X_test, y_test):
    """
    XGBoost 다중클래스 분류 모델을 학습하고 평가 결과를 출력하는 함수.
    라벨이 1~7일 때 자동으로 0~6으로 변환하여 XGBoost에 적용함.
    """

    # --- XGBoost는 label이 0~N-1 필요하므로 변환 ---
    y_train_adj = y_train - 1
    y_test_adj = y_test - 1

    model = XGBClassifier(
        objective='multi:softmax',
        num_class=7,
        eval_metric='mlogloss',
        max_depth=6,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method='hist',
        random_state=42
    )

    print("▶ Training XGBoost...")
    model.fit(X_train, y_train_adj)

    print("▶ Predicting...")
    preds = model.predict(X_test)

    # 다시 1~7 라벨로 복원
    preds_real = preds + 1

    # --- Metrics ---
    acc = accuracy_score(y_test, preds_real)
    macro_f1 = f1_score(y_test, preds_real, average='macro')
    weighted_f1 = f1_score(y_test, preds_real, average='weighted')
    cls_report = classification_report(y_test, preds_real)
    cm = confusion_matrix(y_test, preds_real)

    print("\n===== XGBoost Evaluation =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")
    print("\nClassification Report:")
    print(cls_report)
    print("Confusion Matrix:")
    print(cm)

    return model, preds_real, cm

def train_knn(X_train, y_train, X_test, y_test, k=5):
    # kNN 모델 생성
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # 학습
    knn.fit(X_train, y_train)

    # 예측
    preds = knn.predict(X_test)

    # 평가
    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, preds))
    
    return knn

def train_LightGBM(X_train, y_train, X_test, y_test):
    import lightgbm as lgb
    from sklearn.metrics import mean_squared_error
    
    model = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=300)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    rmse = mean_squared_error(y_test, pred)
    print("Accuracy:", acc)
    print("RMSE:", rmse)
    print("\nClassification Report:\n", classification_report(y_test, pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))

processed_data_folder = './Processed_Data'
output_dir = './results_test'

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
experiments = [
        # Clear - 전체
        ('clear', 'global', 'total')
]
for stroke_type, joint_type, body_part in experiments:
    subjects = ["S00","S01","S02","S03","S05","S06","S07","S10","S17","S23","S08","S09","S12","S13","S15","S16","S21","S22","S14","S19","S20","S24"]
    test_subjects = ["S04","S11","S18"]

    labels = []
    strokes = []
    
    for subj in subjects:
        stroke, skill = dataset.load_subject_data(subj, stroke_type, joint_type, body_part)
        labels.append(int(skill))
        strokes.append(stroke)

    labels_test = []
    strokes_test = []
    for subj in test_subjects:
        stroke, skill = dataset.load_subject_data(subj, stroke_type, joint_type, body_part)
        labels_test.append(int(skill))
        strokes_test.append(stroke)


    ts2vec_model, ts2vec_trainer = TS2VecTrainer.load_model(stroke_type, joint_type, body_part, device=device)
    
    skill_trainer = SkillLevelTrainer(
        ts2vec_model,
        SkillLevelClassifier(embedding_dim=128),
        lr=0.001,
        device=device
    )

    embeddings = skill_trainer.extract_embeddings(strokes)
    embeddings_test = skill_trainer.extract_embeddings(strokes_test)

    labels_expanded = []
    for i, data in enumerate(strokes):
            labels_expanded.extend([labels[i]] * len(data))
    labels_expanded = np.array(labels_expanded, dtype=np.float32)

    labels_test_expanded = []
    for i, data in enumerate(strokes_test):
            labels_test_expanded.extend([labels_test[i]] * len(data))
    labels_test_expanded = np.array(labels_test_expanded, dtype=np.float32)

    train_embeddings, test_embeddings, train_labels_expanded, test_labels_expanded = split_train_test(embeddings, labels_expanded)

    classifier_epochs = 200
    for epoch in range(classifier_epochs):
        loss = skill_trainer.train_epoch(embeddings, labels_expanded, batch_size=64)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{classifier_epochs}, MSE Loss: {loss:.4f}')
    
    # 7. 평가
    results_test = skill_trainer.evaluate(embeddings_test, labels_test_expanded)

    print("=== Test Subject Evaluation ===")
    print(f"Accuracy (rounded): {results_test['accuracy']:.4f}")
    print(f"\npredictions: {results_test['predictions']}")
    print(f"rmse:{results_test['rmse']:.4f}")
    print(results_test['report'])
    print("\nConfusion Matrix (rounded predictions):")
    print(results_test['confusion_matrix'])

    """
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
    
    print(f"\nResults saved to {os.path.join(output_dir, result_filename)}")"""