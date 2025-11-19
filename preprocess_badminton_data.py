import h5py
import pandas as pd
import numpy as np
import os
from datetime import datetime
import json
from scipy import interpolate  # for resampling

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
def timestamp_to_str(unix_time):
    """Unix timestamp를 HDF5 시간 형식으로 변환"""
    return datetime.fromtimestamp(unix_time).strftime('%Y-%m-%d %H:%M:%S.%f').encode('utf-8')

# HDF5 time_str → datetime 변환
def parse_hdf_time(t):
    return datetime.strptime(t.decode("utf-8"), "%Y-%m-%d %H:%M:%S.%f")

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
    
def process_subject(subject_folder, annotation_df, output_dir, forehand_df, backhand_df):
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
    
    hdf_path = os.path.join(subject_folder, hdf_files[0])

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
            target_time_s_high = np.linspace(Forehand_start_time_list[j], Forehand_stop_time_list[j],
                                             num=150, endpoint=True)

            fn_local = interpolate.interp1d(local_time_s, local_data, axis=0, kind='slinear', fill_value='extrapolate')
            fn_global = interpolate.interp1d(global_time_s, global_data, axis=0, kind='slinear', fill_value='extrapolate')

            local_resampled = fn_local(target_time_s_high)
            global_resampled = fn_global(target_time_s_high)

            local_clear_np.append(local_resampled)
            global_clear_np.append(global_resampled)

        local_clear_np = np.array(local_clear_np)
        global_clear_np = np.array(global_clear_np)

        local_mean = np.mean(local_clear_np, axis=0)
        local_std = np.std(local_clear_np, axis=0)
        global_mean = np.mean(global_clear_np, axis=0)
        global_std = np.std(global_clear_np, axis=0)

        local_clear_np = (local_clear_np - local_mean) / local_std
        global_clear_np = (global_clear_np - global_mean) / global_std

        save_dir = os.path.join('./Processed_Data', subject_id_short, f'{subject_id_short}_clear')
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'local.npy'), local_clear_np)
        np.save(os.path.join(save_dir, 'global.npy'), global_clear_np)

    # ==========================================================================================
    #                                      DRIVE 처리
    # ==========================================================================================
    with h5py.File(hdf_path, 'r') as file_data:
        local_time_s = np.squeeze(np.array(file_data['pns-joint']['local-position']['time_s']))
        local_data = np.squeeze(np.array(file_data['pns-joint']['local-position']['data'])) 
        global_time_s = np.squeeze(np.array(file_data['pns-joint']['global-position']['time_s']))
        global_data = np.squeeze(np.array(file_data['pns-joint']['global-position']['data'])) 

        local_drive_np = []
        global_drive_np = []

        for j in range(len(Backhand_start_time_list)):
            target_time_s_high = np.linspace(Backhand_start_time_list[j], Backhand_stop_time_list[j],
                                             num=150, endpoint=True)

            fn_local = interpolate.interp1d(local_time_s, local_data, axis=0, kind='slinear', fill_value='extrapolate')
            fn_global = interpolate.interp1d(global_time_s, global_data, axis=0, kind='slinear', fill_value='extrapolate')

            local_resampled = fn_local(target_time_s_high)
            global_resampled = fn_global(target_time_s_high)

            local_drive_np.append(local_resampled)
            global_drive_np.append(global_resampled)

        local_drive_np = np.array(local_drive_np)
        global_drive_np = np.array(global_drive_np)

        local_mean = np.mean(local_drive_np, axis=0)
        local_std = np.std(local_drive_np, axis=0)
        global_mean = np.mean(global_drive_np, axis=0)
        global_std = np.std(global_drive_np, axis=0)

        local_drive_np = (local_drive_np - local_mean) / local_std
        global_drive_np = (global_drive_np - global_mean) / global_std

        save_dir = os.path.join('./Processed_Data', subject_id_short, f'{subject_id_short}_drive')
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'local.npy'), local_drive_np)
        np.save(os.path.join(save_dir, 'global.npy'), global_drive_np)

    print(f"clear num: {len(local_clear_np)}, drive num: {len(local_drive_np)}")


def main():
    annotation_file = './Documentations/Annotation Data File.xlsx'
    data_root = './Data_Archive'
    output_dir = './Processed_Data'

    print("Loading annotation file...")
    df = pd.read_excel(annotation_file)

    df_subject_forehand = df[df["Annotation Level 1\n(Stroke Type)"] == 'Forehand Clear']
    df_subject_backhand = df[df["Annotation Level 1\n(Stroke Type)"] == 'Backhand Driving']
    

    print(f"Total annotations: {len(df)}")
    os.makedirs(output_dir, exist_ok=True)

    for i in range(25):
        subject_folder = os.path.join(data_root, f"Sub{i:02d}")
        if os.path.exists(subject_folder):
            process_subject(subject_folder, df, output_dir,df_subject_forehand, df_subject_backhand)

    print("\n" + "=" * 60)
    print("Preprocessing completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
    #process_badminton_skill_levels('./Documentations/Skill Level Annotation Detail File.xlsx')
