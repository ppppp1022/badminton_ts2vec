import h5py
import pandas as pd
import numpy as np
import os
from datetime import datetime
import json
from scipy.interpolate import interp1d

def load_skill_levels_from_annotation(annotation_filepath):
    """
    Load skill levels and subject groups from annotation JSON file.
    """
    with open(annotation_filepath, 'r') as f:
        data = json.load(f)

    # Load clear & drive skill levels
    clear_skills = data.get("clear_skill_level", [])
    drive_skills = data.get("drive_skill_level", [])

    # Load subject groups
    subject_groups = {
        "beginner": data.get("beginner_subjects", []),
        "intermediate": data.get("intermediate_subjects", []),
        "expert": data.get("expert_subjects", [])
    }

    return clear_skills, drive_skills, subject_groups
def convert_bytes_to_numeric_time(time_bytes_array):
    """
    numpy.bytes_ 형태의 시간 배열을 받아서 float 형태의 Unix Timestamp로 변환합니다.
    
    Args:
        time_bytes_array (np.ndarray): b'2023-02-16...' 형태의 배열
        
    Returns:
        np.ndarray: 초 단위의 실수형(float) 타임스탬프 배열
    """
    # 1. Bytes -> String 디코딩 (Vectorized)
    # numpy.char.decode를 사용하면 배열 전체를 한 번에 utf-8 문자열로 바꿉니다.
    time_str_decoded = np.char.decode(time_bytes_array, 'utf-8')
    
    # 2. String -> Datetime 변환 (Pandas가 가장 빠르고 정확함)
    dt_index = pd.to_datetime(time_str_decoded)
    
    # 3. Datetime -> Numeric (Float, Unix Timestamp) 변환
    # pandas datetime은 내부적으로 nanosecond(10^-9) 정수로 저장됩니다.
    # 이를 초(second) 단위로 바꾸기 위해 10^9로 나눕니다.
    numeric_time = dt_index.astype(np.int64) / 10**9
    
    return numeric_time.to_numpy() # 다시 numpy array로 반환

# HDF5 time_str → datetime 변환
def parse_hdf_time(t):
    return datetime.strptime(t.decode("utf-8"), "%Y-%m-%d %H:%M:%S.%f")

def resize_scipy(data, target_length):
    N = data.shape[0]
    
    old_x = np.linspace(0, N - 1, N)
    new_x = np.linspace(0, N - 1, target_length)
    
    # axis=0을 기준으로 보간 함수 생성
    f = interp1d(old_x, data, kind='linear', axis=0)
    
    return f(new_x)
# datetime 리스트에서 이진 탐색
def timeSearch(time_list, target_time):
    left, right = 0, len(time_list) - 1

    while left <= right:
        mid = (left + right) // 2

        # 마지막 index일 경우 바로 mid 반환
        if mid == len(time_list) - 1:
            return mid

        # 정상 비교
        if time_list[mid] <= target_time < time_list[mid + 1]:
            return mid
        elif target_time < time_list[mid]:
            right = mid - 1
        else:
            left = mid + 1

    # 못 찾으면 마지막 index 반환
    return len(time_list) - 1

def process_badminton_skill_levels(annotation_filepath):
    
    data = pd.read_excel(annotation_filepath)
    clear_skill_level = data.get("Unnamed: 8", [])
    drive_skill_level = data.get("Unnamed: 16", [])

def process_subject(subject_folder, forehand_df, backhand_df, order_forehand):
    subject_id = os.path.basename(subject_folder)  # Sub00
    subject_id_short = 'S' + subject_id[3:]        # S00

    forehand_subject_id_df = forehand_df[forehand_df['Subject Number']==subject_id]
    backhand_subject_id_df = backhand_df[backhand_df['Subject Number']==subject_id]

    Forehand_start_time_list = forehand_subject_id_df['Annotation Start Time'].values.tolist()
    Forehand_stop_time_list = forehand_subject_id_df['Annotation Stop Time'].values.tolist()
    Backhand_start_time_list = backhand_subject_id_df['Annotation Start Time'].values.tolist()
    Backhand_stop_time_list = backhand_subject_id_df['Annotation Stop Time'].values.tolist()


    hdf_files = [f for f in os.listdir(subject_folder) if f.endswith('.hdf5')]
    if not hdf_files:
        return
    print(f"\nProcessing {subject_id}...")
    hdf_file = hdf_files[order_forehand]
    hdf_path = os.path.join(subject_folder, hdf_file)
    # ==========================================================================================
    #                                      CLEAR 처리
    # ==========================================================================================
    
    with h5py.File(hdf_path, 'r') as file_data:
        local_time_s = np.squeeze(np.array(file_data['pns-joint']['local-position']['time_s']))
        local_data = np.squeeze(np.array(file_data['pns-joint']['local-position']['data'])) 
        
        global_time_s = np.squeeze(np.array(file_data['pns-joint']['global-position']['time_s']))
        global_data = np.squeeze(np.array(file_data['pns-joint']['global-position']['data']))
        
        local_clear_np = []
        global_clear_np = []

        for j in range(len(Forehand_start_time_list)):
            time_start_index = timeSearch(local_time_s, Forehand_start_time_list[j])
            time_stop_index = timeSearch(local_time_s, Forehand_stop_time_list[j])
            local_position = local_data[time_start_index:time_stop_index]
            global_position = global_data[time_start_index:time_stop_index]

            local_resampled = resize_scipy(local_position, 300)
            global_resampled = resize_scipy(global_position, 300)

            data_np = np.array(local_resampled)
            means = np.mean(data_np, axis=0)
            stds = np.std(data_np, axis=0)
            local_normalized_data = (data_np - means) / (stds + 1e-8)

            data_np = np.array(global_resampled)
            means = np.mean(data_np, axis=0)
            stds = np.std(data_np, axis=0)
            global_normalized_data = (data_np - means) / (stds + 1e-8)

            pivot_local = local_normalized_data[0][0:3]
            pivot_global = global_normalized_data[0][0:3]

            repeat_count = local_resampled.shape[1] // 3
            full_pivot_local = np.tile(pivot_local, repeat_count)
            full_pivot_global = np.tile(pivot_global, repeat_count)

            local_normalized_data -= full_pivot_local
            global_normalized_data -= full_pivot_global

            local_clear_np.append(local_normalized_data)
            global_clear_np.append(global_normalized_data)

        save_dir = os.path.join('./Processed_Data', subject_id_short, f'{subject_id_short}_clear')
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'local.npy'), local_clear_np)
        np.save(os.path.join(save_dir, 'global.npy'), global_clear_np)

    # ==========================================================================================
    #                                      DRIVE 처리
    # ==========================================================================================
    order = (order_forehand+1)%2
    hdf_file = hdf_files[order]
    hdf_path = os.path.join(subject_folder, hdf_file)
    
    with h5py.File(hdf_path, 'r') as file_data:
        local_time_s = np.squeeze(np.array(file_data['pns-joint']['local-position']['time_s']))
        local_data = np.squeeze(np.array(file_data['pns-joint']['local-position']['data'])) 
        global_time_s = np.squeeze(np.array(file_data['pns-joint']['global-position']['time_s']))
        global_data = np.squeeze(np.array(file_data['pns-joint']['global-position']['data'])) 

        local_drive_np = []
        global_drive_np = []

        for j in range(len(Backhand_start_time_list)):
            time_start_index = timeSearch(local_time_s, Backhand_start_time_list[j])
            time_stop_index = timeSearch(local_time_s, Backhand_stop_time_list[j])
 
            local_position = local_data[time_start_index:time_stop_index]
            global_position = global_data[time_start_index:time_stop_index]

            if len(local_position)==0:
                continue
            local_resampled = resize_scipy(local_position, 300)
            global_resampled = resize_scipy(global_position, 300)

            data_np = np.array(local_resampled)
            means = np.mean(data_np, axis=0)
            stds = np.std(data_np, axis=0)
            local_normalized_data = (data_np - means) / (stds + 1e-8)

            data_np = np.array(global_resampled)
            means = np.mean(data_np, axis=0)
            stds = np.std(data_np, axis=0)
            global_normalized_data = (data_np - means) / (stds + 1e-8)

            pivot_local = local_normalized_data[0][0:3]
            pivot_global = global_normalized_data[0][0:3]

            repeat_count = local_resampled.shape[1] // 3
            full_pivot_local = np.tile(pivot_local, repeat_count)
            full_pivot_global = np.tile(pivot_global, repeat_count)

            local_normalized_data -= full_pivot_local
            global_normalized_data -= full_pivot_global

            local_drive_np.append(local_normalized_data)
            global_drive_np.append(global_normalized_data)


        save_dir = os.path.join('./Processed_Data', subject_id_short, f'{subject_id_short}_drive')
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'local.npy'), local_drive_np)
        np.save(os.path.join(save_dir, 'global.npy'), global_drive_np)
    
    #print(f"clear num: {len(local_clear_np)}, drive num: {len(local_drive_np)}")


def main():
    annotation_file = './Documentations/Annotation Data File.xlsx'
    data_root = './Data_Archive'
    output_dir = './Processed_Data'

    order_forehand = [0,0,1,0,1,1,1,1,1,1,0,0,1,0,1,1,0,1,1,0,0,1,0,0,1]

    print("Loading annotation file...")
    df = pd.read_excel(annotation_file)

    df_subject_forehand = df[df["Annotation Level 1\n(Stroke Type)"] == 'Forehand Clear']
    df_subject_backhand = df[df["Annotation Level 1\n(Stroke Type)"] == 'Backhand Driving']
    

    print(f"Total annotations: {len(df)}")
    os.makedirs(output_dir, exist_ok=True)

    for i in range(25):
        subject_folder = os.path.join(data_root, f"Sub{i:02d}")
        if os.path.exists(subject_folder):
            process_subject(subject_folder ,df_subject_forehand, df_subject_backhand, order_forehand[i])

    print("\n" + "=" * 60)
    print("Preprocessing completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
    #process_badminton_skill_levels('./Documentations/Skill Level Annotation Detail File.xlsx')
