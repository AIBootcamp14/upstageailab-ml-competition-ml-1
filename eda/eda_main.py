# EDA 프로세스의 시작점

# eda/eda_main.py 

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# --- 작업 디렉토리 설정 부분 ---
# 현재 파일 (eda_main.py)의 절대 경로를 기준으로 프로젝트 루트를 찾음
# eda_main.py -> eda/ -> project_root (AIBootcamp14-ml-project1-kimyoung9689-fork)
project_git_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 실제 작업 디렉토리 (project1)
project_actual_working_dir = os.path.join(project_git_root, 'project1') 

# 작업 디렉토리 변경
os.chdir(project_actual_working_dir)
print(f"현재 작업 디렉토리: {os.getcwd()}") # 변경 후 작업 디렉토리 출력
print(f"작업 디렉토리 변경 완료: {os.getcwd()}")

# 데이터 디렉토리 설정 (변경된 작업 디렉토리 기준 상대 경로)
# 이제 DATA_DIR은 'project1/data'가 된다.
DATA_DIR = 'data' 
print(f"데이터 디렉토리: {os.path.abspath(DATA_DIR)}")
# --- 작업 디렉토리 설정 끝 ---

# 프로젝트 Git 루트 디렉토리를 sys.path에 추가 (모듈 import를 위해 여전히 필요)
# eda.eda_utils, eda.visualization 등을 임포트할 때 필요
if project_git_root not in sys.path:
    sys.path.append(project_git_root)

# eda 폴더 안의 모듈들을 import
# 이 부분은 이전과 동일하게 유지
from eda.eda_utils import load_datasets, find_target_column
from eda.geo_feature_engineering import create_features_for_eda
from eda.visualization import perform_eda_on_features 

# --- 메인 실행 로직은 이전과 동일하게 유지 ---
if __name__ == "__main__":
    print("\n--- 지리적 피처 EDA 스크립트 시작 ---")
    
    # 2.1. 필요한 모든 데이터셋 로드
    datasets = load_datasets(DATA_DIR)

    if datasets is None or 'train' not in datasets or datasets['train'] is None:
        print("데이터 로드 실패 또는 'train' 데이터셋을 찾을 수 없어. EDA를 종료할게.")
    else:
        train_df_for_eda = datasets['train'].copy()

        print("새로운 지리적 피처 생성 중...")
        enhanced_train_for_eda = create_features_for_eda(
            train_df_for_eda,
            academy_df=datasets['academy'],
            traffic_df=datasets['traffic'],
            bus_df=datasets['bus'],
            subway_df=datasets['subway'],
            slope_df=datasets['slope']
        )
        print("모든 지리적 피처 생성 완료.")

        target_col = find_target_column(enhanced_train_for_eda)

        if target_col:
            perform_eda_on_features(enhanced_train_for_eda, target_col)
        else:
            print("타겟 컬럼을 찾을 수 없어. EDA를 수행하지 않을게.")

    print("\n--- 지리적 피처 EDA 스크립트 종료 ---")