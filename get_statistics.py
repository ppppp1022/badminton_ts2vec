import json
import pandas as pd
import numpy as np

# 1. 파일 경로 설정 (이 부분을 본인의 파일 경로로 수정하세요)
MODEL_NAME = 'convlstm'
FILE_PATH = f'./results_{MODEL_NAME}/all_results_summary.json'

def calculate_metrics(cm):
    cm = np.array(cm)
    n_classes = cm.shape[0]
    precisions, recalls, f1_scores = [], [], []
    
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        
    return precisions, recalls, f1_scores

# 데이터 로드
with open(FILE_PATH, 'r') as f:
    data = json.load(f)

summary_list = []
detailed_list = []

for exp_data in data:
    exp_name = exp_data['experiment']
    cms = exp_data['confusion_matrix']
    fold_precisions, fold_recalls, fold_f1s = [], [], []
    
    for fold_idx, cm in enumerate(cms):
        p, r, f1 = calculate_metrics(cm)
        # Macro-average (클래스별 지표의 단순 평균)
        fold_precisions.append(np.mean(p))
        fold_recalls.append(np.mean(r))
        fold_f1s.append(np.mean(f1))
        
        # 상세 데이터 저장 (폴드별/클래스별)
        for c_idx in range(len(p)):
            detailed_list.append({
                'experiment': exp_name, 'fold': fold_idx, 'class': c_idx,
                'precision': p[c_idx], 'recall': r[c_idx], 'f1_score': f1[c_idx]
            })
            
    # 실험별 요약 정보 계산
    summary_list.append({
        'experiment': exp_name,
        'mean_accuracy': exp_data['fold_accuracies'],
        'mean_precision': np.mean(fold_precisions) * 100,
        'mean_recall': np.mean(fold_recalls) * 100,
        'mean_f1': np.mean(fold_f1s) * 100,
        'std_f1': np.std(fold_f1s) * 100
    })

# 결과 저장
df_summary = pd.DataFrame(summary_list)
df_details = pd.DataFrame(detailed_list)

df_summary.to_csv(f'./results_{MODEL_NAME}/experiment_summary.csv', index=False)
df_details.to_csv(f'./results_{MODEL_NAME}/detailed_metrics.csv', index=False)

print("분석이 완료되었습니다. 'experiment_summary.csv' 파일을 확인하세요.")
print(df_summary)