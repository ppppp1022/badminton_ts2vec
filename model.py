import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, classification_report
from Networks.SciDataModels import Conv1DRefinedModel, ConvLSTMRefinedModel, LSTMRefinedModel, TransformerRefinedModel

def pad_or_truncate(data_list: List[np.ndarray], target_length: int = 200) -> np.ndarray:
    """
    모든 stroke를 동일한 길이로 맞춤
    
    Args:
        data_list: List of (seq_len, features) arrays
        target_length: 목표 시퀀스 길이
    
    Returns:
        padded_data: (num_strokes, target_length, features)
    """
    processed = []
    
    for stroke in data_list:
        seq_len, features = stroke.shape
        
        if seq_len > target_length:
            # Truncate (앞에서 자르기)
            processed.append(stroke[:target_length])
        elif seq_len < target_length:
            # Pad (0으로 채우기)
            padding = np.zeros((target_length - seq_len, features))
            processed.append(np.vstack([stroke, padding]))
        else:
            processed.append(stroke)
    
    return np.array(processed) 

# ============================================================================
# SciData 모델 Trainer
# ============================================================================

class SciDataModelTrainer:
    """Conv1D, LSTM, ConvLSTM, Transformer 모델 학습"""
    
    def __init__(self, model, model_name: str, lr=0.001, device='cuda', seq_length=200):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.seq_length = seq_length  # 고정 시퀀스 길이
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
    
    def pad_or_truncate(self, strokes: List[np.ndarray]) -> np.ndarray:
        """모든 stroke를 동일한 길이로"""
        processed = []
        for stroke in strokes:
            seq_len, features = stroke.shape
            
            if seq_len > self.seq_length:
                processed.append(stroke[:self.seq_length])
            elif seq_len < self.seq_length:
                padding = np.zeros((self.seq_length - seq_len, features))
                processed.append(np.vstack([stroke, padding]))
            else:
                processed.append(stroke)
        
        return np.array(processed, dtype=np.float32)
        
    def train_epoch(self, data_list: List[np.ndarray], labels: List[int], batch_size=32):
        """1 에폭 학습"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # 모든 stroke를 하나의 배열로
        all_strokes = []
        all_labels = []
        
        for subject_idx, subject_data in enumerate(data_list):
            for stroke in subject_data:
                all_strokes.append(stroke)
                all_labels.append(labels[subject_idx])
        
        # Pad or truncate
        all_strokes = self.pad_or_truncate(all_strokes)
        all_labels = np.array(all_labels, dtype=int)
        
        # Shuffle
        indices = np.random.permutation(len(all_strokes))
        all_strokes = all_strokes[indices]
        all_labels = all_labels[indices]
        
        # Batch 학습
        for i in range(0, len(all_strokes), batch_size):
            batch_x = all_strokes[i:i+batch_size]
            batch_y = all_labels[i:i+batch_size]
            
            if len(batch_x) < 2:
                continue
            
            # Tensor 변환
            batch_x_tensor = torch.FloatTensor(batch_x).to(self.device)
            batch_y_tensor = torch.LongTensor(batch_y - 1).to(self.device)
            
            # Forward
            outputs = self.model(batch_x_tensor)
            loss = self.criterion(outputs, batch_y_tensor)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def evaluate(self, data_list: List[np.ndarray], labels: List[int]) -> Dict:
        """모델 평가"""
        self.model.eval()
        
        # 모든 stroke 수집
        all_strokes = []
        all_labels = []
        
        for subject_idx, subject_data in enumerate(data_list):
            for stroke in subject_data:
                all_strokes.append(stroke)
                all_labels.append(labels[subject_idx])
        
        # Pad or truncate
        all_strokes = self.pad_or_truncate(all_strokes)
        all_labels = np.array(all_labels, dtype=int)
        
        # Prediction
        with torch.no_grad():
            strokes_tensor = torch.FloatTensor(all_strokes).to(self.device)
            outputs = self.model(strokes_tensor)
            predictions = outputs.argmax(dim=1).cpu().numpy() + 1
        
        # Metrics
        accuracy = accuracy_score(all_labels, predictions)
        
        # RMSE
        mse = np.mean((predictions - all_labels) ** 2)
        rmse = np.sqrt(mse)
        
        # R² Score
        r2 = r2_score(all_labels, predictions)
        
        # Classification report
        unique_labels = sorted(set(all_labels) | set(predictions))
        target_names = [f'Level {i}' for i in unique_labels]
        
        report = classification_report(
            all_labels,
            predictions,
            labels=unique_labels,
            target_names=target_names,
            zero_division=0
        )
        
        cm = confusion_matrix(all_labels, predictions, labels=unique_labels)
        
        return {
            'accuracy': float(accuracy),
            'rmse': float(rmse),
            'r2_score': float(r2),
            'predictions': predictions,
            'report': report,
            'confusion_matrix': cm
        }
    
    def save_model(self, stroke_type: str, joint_type: str, body_part: str, 
                   model_config: dict = None, save_dir='./TrainedModels'):
        """
        모델 저장 (설정 포함)
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        model_name = f"{self.model_name}_{stroke_type}_{joint_type}_{body_part}.pth"
        model_path = os.path.join(save_dir, model_name)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_name': self.model_name,
            'stroke_type': stroke_type,
            'joint_type': joint_type,
            'body_part': body_part,
            'model_config': model_config  # 모델 생성 파라미터 저장
        }
        
        torch.save(checkpoint, model_path)
        print(f"Model saved to {model_path}")
        
        return model_path
    
    @staticmethod
    def load_model(model_class, model_name: str, stroke_type: str, joint_type: str, 
                   body_part: str, input_dim: int, device='cuda', save_dir='./TrainedModels'):
        """
        저장된 모델 로드
        
        Args:
            model_class: Conv1DRefinedModel, LSTMRefinedModel 등
            model_name: 'conv1d', 'lstm', 'convlstm', 'transformer'
            input_dim: 입력 차원
            ... (나머지 인자)
        """
        import os
        
        model_filename = f"{model_name}_{stroke_type}_{joint_type}_{body_part}.pth"
        model_path = os.path.join(save_dir, model_filename)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=device)
        
        # 모델 재생성
        num_classes = 7
        
        if model_name == 'conv1d':
            model = Conv1DRefinedModel(input_channels=input_dim, output_size=num_classes)
        elif model_name == 'lstm':
            model = LSTMRefinedModel(input_size=input_dim, hidden_size=64, 
                                     num_layers=2, output_size=num_classes)
        elif model_name == 'convlstm':
            model = ConvLSTMRefinedModel(input_channels=input_dim, hidden_size=64, 
                                         output_size=num_classes)
        elif model_name == 'transformer':
            model = TransformerRefinedModel(input_channels=input_dim, embed_size=64, 
                                           output_size=num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        trainer = SciDataModelTrainer(model, model_name, device=device)
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Model loaded from {model_path}")
        
        return model, trainer



# ============================================================================
# 메인 파이프라인
# ============================================================================

def train_scidata_models(dataset, stroke_type, joint_type, body_part, device='cuda'):
    """
    여러 SciData 모델 학습
    
    Args:
        dataset: ProcessedBadmintonDataset
        stroke_type: 'clear' or 'drive'
        joint_type: 'local' or 'global'
        body_part: 'arm', 'leg', 'total'
        device: 'cuda' or 'cpu'
    """
    
    print("\n" + "="*80)
    print(f"Training SciData Models: {stroke_type} - {joint_type} - {body_part}")
    print("="*80)
    
    # 데이터 분할
    train_subjects, test_subjects, train_labels, test_labels = dataset.split_data_by_skill(stroke_type)
    
    print(f"\nTrain subjects ({len(train_subjects)}): {train_subjects}")
    print(f"Test subjects ({len(test_subjects)}): {test_subjects}")
    
    # 데이터 로드
    train_data_list = []
    for subject in train_subjects:
        data, skill = dataset.load_subject_data(subject, stroke_type, joint_type, body_part)
        train_data_list.append(data)
    
    test_data_list = []
    for subject in test_subjects:
        data, skill = dataset.load_subject_data(subject, stroke_type, joint_type, body_part)
        test_data_list.append(data)
    
# Input dimension
    sample_data = train_data_list[0][0]
    seq_len, input_dim = sample_data.shape
    print(f"\nOriginal data shape: ({seq_len}, {input_dim})")
    
    # 고정 시퀀스 길이 설정
    target_seq_len = 200
    print(f"Target sequence length: {target_seq_len}")
    
    num_classes = 7
    
    # 모델 정의 - Transformer는 다른 인자 사용
    models_config = {
        'conv1d': lambda: Conv1DRefinedModel(
            input_channels=input_dim,
            hidden_features=64,
            output_size=num_classes
        ),
        'lstm': lambda: LSTMRefinedModel(
            input_channels=input_dim,
            hidden_features=64,
            output_size=num_classes
        ),
        'convlstm': lambda: ConvLSTMRefinedModel(
            input_channels=input_dim,
            hidden_features=64,
            output_size=num_classes
        ),
        # Transformer는 다른 인자명 사용할 수 있음
        'transformer': lambda: TransformerRefinedModel(
            input_channels=input_dim,
            embed_size=64,  # 또는 hidden_features 대신
            output_size=num_classes
        )
    }
    
    results = {}
    
    # 각 모델 학습
    for model_name, model_fn in models_config.items():
        print("\n" + "="*60)
        print(f"Training {model_name.upper()} Model")
        print("="*60)
        
        try:
            # 모델 생성
            model = model_fn()
            trainer = SciDataModelTrainer(
                model, 
                model_name, 
                lr=0.001, 
                device=device,
                seq_length=target_seq_len  # 고정 길이 전달
            )
            
            # 학습
            print("\nTraining...")
            for epoch in range(50):
                loss = trainer.train_epoch(train_data_list, train_labels, batch_size=32)
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch {epoch+1}/50, Loss: {loss:.4f}')
            
            # 평가
            print("\nEvaluating...")
            eval_results = trainer.evaluate(test_data_list, test_labels)
            
            print(f"\nResults for {model_name.upper()}:")
            print(f"  Accuracy: {eval_results['accuracy']:.4f}")
            print(f"  RMSE: {eval_results['rmse']:.4f}")
            print(f"  R² Score: {eval_results['r2_score']:.4f}")
            
            # 모델 저장
            model_config = {
                'input_dim': input_dim,
                'seq_length': target_seq_len,
                'num_classes': num_classes
            }
            trainer.save_model(stroke_type, joint_type, body_part, model_config=model_config)
            
            results[model_name] = eval_results
            
        except Exception as e:
            print(f"\nError training {model_name}: {e}")
            import traceback
            traceback.print_exc()
    # 결과 비교
    print("\n" + "="*80)
    print("Model Comparison")
    print("="*80)
    
    for model_name, result in results.items():
        print(f"{model_name.upper():15s} - Accuracy: {result['accuracy']:.4f}, "
              f"RMSE: {result['rmse']:.4f}, R²: {result['r2_score']:.4f}")
    
    # 최고 성능 모델
    if results:
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\nBest Model: {best_model[0].upper()} (Accuracy: {best_model[1]['accuracy']:.4f})")
    
    return results


# ============================================================================
# 통합 실행
# ============================================================================

def main_compare_all_models():
    """TS2Vec + SciData 모델 모두 비교"""
    
    from processed_data_loader import ProcessedBadmintonDataset
    from preprocess_badminton_data import load_skill_levels_from_annotation
    
    # 데이터셋 설정
    dataset = ProcessedBadmintonDataset('./Processed_Data')
    clear_skills, drive_skills, subject_groups = load_skill_levels_from_annotation('./configs/skill_levels.json')
    
    dataset.clear_skill_level = clear_skills
    dataset.drive_skill_level = drive_skills
    dataset.beginner_subjects = subject_groups['beginner']
    dataset.intermediate_subjects = subject_groups['intermediate']
    dataset.expert_subjects = subject_groups['expert']
    
    # Joint 인덱스 설정
    dataset.local_arm_index = [13, 14, 15, 16, 17, 18, 19, 20]
    dataset.global_arm_index = [13, 14, 15, 16, 17, 18, 19, 20]
    dataset.local_leg_index = [1, 2, 3, 4, 5, 6]
    dataset.global_leg_index = [1, 2, 3, 4, 5, 6]
    dataset.local_total_index = list(range(21))
    dataset.global_total_index = list(range(21))
    
    # 실험 설정
    stroke_type = 'clear'
    joint_type = 'global'
    body_part = 'arm'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # SciData 모델들 학습
    scidata_results = train_scidata_models(dataset, stroke_type, joint_type, body_part, device)
    
    print("\n" + "="*80)
    print("All Experiments Completed!")
    print("="*80)


if __name__ == '__main__':
    main_compare_all_models()