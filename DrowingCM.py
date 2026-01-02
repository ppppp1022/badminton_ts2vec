import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_and_save_confusion_matrices(json_file_path, output_dir):
    """
    JSON 결과 파일을 읽어 Confusion Matrix 그래프를 생성하고 저장합니다.
    
    Args:
        json_file_path (str): 읽어올 JSON 파일 경로 (예: './results/summary.json')
        output_dir (str): 그래프를 저장할 폴더 경로 (예: './plots')
    """
    
    # 1. 저장 폴더 생성 (없으면 만듦)
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. JSON 파일 로드
    if not os.path.exists(json_file_path):
        print(f"Error: 파일이 존재하지 않습니다 -> {json_file_path}")
        return

    with open(json_file_path, 'r', encoding='utf-8') as f:
        try:
            results_data = json.load(f)
            print(f"Successfully loaded data from {json_file_path}")
        except json.JSONDecodeError:
            print("Error: JSON 파일 형식이 올바르지 않습니다.")
            return

    # 3. 클래스 라벨 정의 (7단계)
    class_labels = [f'Lvl {i}' for i in range(1, 8)]

    # 4. 각 실험별로 순회하며 그래프 생성 및 저장
    print(f"\nGenerating plots in '{output_dir}'...")
    
    for i, exp_data in enumerate(results_data):
        exp_name = exp_data.get('experiment', f'experiment_{i}')
        fold_matrices = np.array(exp_data['confusion_matrix'])
        accuracy = exp_data.get('fold_accuracies', 0.0)
        
        # 5-Fold Matrix 합산
        total_cm = np.sum(fold_matrices, axis=0)
        
        # 그래프 설정
        plt.figure(figsize=(10, 8))
        
        # Heatmap 그리기
        sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_labels, yticklabels=class_labels,
                    annot_kws={"size": 12}) # 숫자 크기
        
        # 타이틀 및 라벨
        plt.title(f"Experiment: {exp_name}\nAvg Accuracy: {accuracy:.2f}%", fontsize=15, fontweight='bold', pad=20)
        plt.ylabel('True Class', fontsize=12)
        plt.xlabel('Predicted Class', fontsize=12)
        
        # 5. 파일로 저장 (Export)
        # 파일명에 실험 이름을 넣어 구분
        filename = f"{exp_name}_confusion_matrix.png"
        save_path = os.path.join(output_dir, filename)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight') # 고해상도 저장
        plt.close() # 메모리 해제 (중요)
        
        print(f"Saved: {filename}")

    print(f"\nAll plots saved successfully to {output_dir}")

# ==========================================
# 실행 부분
# ==========================================
if __name__ == "__main__":
    # 1. 읽어올 파일 경로 (사용자의 JSON 파일 위치로 수정하세요)
    # 예: 아까 저장 로직이 './results_2d_convlstm/all_results_summary.json' 이라면 그 경로 입력
    INPUT_JSON_PATH = './results_ts2vec/all_results_summary.json' 
    
    # 2. 그래프를 내보낼 폴더
    OUTPUT_PLOT_DIR = './result_plots'
    # 폴더가 없으면 만들고 더미 파일 저장
    if not os.path.exists(os.path.dirname(INPUT_JSON_PATH)):
        os.makedirs(os.path.dirname(INPUT_JSON_PATH))

    # 함수 실행
    plot_and_save_confusion_matrices(INPUT_JSON_PATH, OUTPUT_PLOT_DIR)