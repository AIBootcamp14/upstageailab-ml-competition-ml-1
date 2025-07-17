# 데이터 로드와 기본적인 전처리 기능 담당

# eda/eda_utils.py

import os
import pandas as pd
import numpy as np
import warnings # warnings 모듈 import 추가
warnings.filterwarnings('ignore') # 경고 메시지 무시 설정 추가

# --- 데이터 로드 함수 ---
def load_csv(filepath):
    """CSV 파일을 안전하게 로드. 파일이 없으면 경고 출력."""
    try:
        if os.path.exists(filepath):
            print(f"'{filepath}' 파일 로드 중...")
            return pd.read_csv(filepath, encoding='utf-8-sig')
        else:
            print(f"경고: '{filepath}' 파일이 없어.")
            return None
    except Exception as e:
        print(f"오류: '{filepath}' 파일 로드 중 에러 발생: {e}")
        return None

def load_datasets(data_dir):
    """프로젝트에 필요한 모든 데이터셋을 로드."""
    datasets = {}
    
    # 메인 훈련 데이터셋 (train5.csv 사용)
    datasets['train'] = load_csv(os.path.join(data_dir, 'train5.csv'))
    
    # 지리적 정보 관련 데이터셋
    datasets['academy'] = load_csv(os.path.join(data_dir, 'processed_academy.csv'))
    datasets['traffic'] = load_csv(os.path.join(data_dir, 'processed_traffic.csv'))
    datasets['bus'] = load_csv(os.path.join(data_dir, 'bus_feature.csv'))
    datasets['subway'] = load_csv(os.path.join(data_dir, 'subway_feature.csv'))
    
    # 등고선 데이터 로드 - 이름을 'slope'로 변경하여 eda_main.py와 일치시킴
    datasets['slope'] = load_contour_data(data_dir) # 여기가 핵심 변경점!
    if datasets['slope'] is None:
        print(f"경고: 'slope' (등고선) 데이터셋 로드 실패.")
    
    # 데이터 로드 성공 여부 확인
    for name, df in datasets.items():
        if df is None:
            # 'train' 데이터가 없으면 EDA 진행 불가
            if name == 'train': 
                print(f"경고: '{name}' 데이터셋 로드 실패. 필수 'train' 데이터 없으면 진행 불가.")
                return None
            # 다른 데이터는 없어도 진행은 가능하지만 경고 출력
            else:
                print(f"경고: '{name}' 데이터셋이 없어 지리적 피처 생성에 영향을 줄 수 있어.")
                
    return datasets

def load_contour_data(data_dir):
    """등고선 데이터를 찾아 로드하고 필요한 컬럼만 정리."""
    # os.listdir을 사용하기 전에 data_dir이 실제 존재하는지 확인
    if not os.path.exists(data_dir):
        print(f"오류: 데이터 디렉토리 '{data_dir}'를 찾을 수 없어.")
        return None

    contour_files = []
    try:
        files_in_dir = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        for file in files_in_dir:
            # 파일 이름에 등고선 관련 키워드가 있는지 확인
            if any(kw.lower() in file.lower() for kw in ['등고선', '경사도', '표고', 'contour', 'elevation', 'geodata', 'knn']): # 'knn' 추가
                filepath = os.path.join(data_dir, file)
                contour_files.append((filepath, os.path.getsize(filepath) / (1024 * 1024))) 
        contour_files.sort(key=lambda x: x[1], reverse=True) # 파일 크기 기준으로 정렬

    except Exception as e:
        print(f"등고선 파일 목록 스캔 중 오류 발생: {e}")
        return None

    if not contour_files:
        print(f"등고선 파일을 찾을 수 없어: {os.path.join(data_dir, 'processed_서울시경사도5000(geodata)_KNN보간.csv')} (기타 등고선 관련 파일 포함)")
        return None
    
    # 가장 큰 등고선 파일 하나만 로드
    file_path_to_load = contour_files[0][0]
    print(f"EDA용 등고선 파일 '{os.path.basename(file_path_to_load)}' 로드 중...") # 파일명만 출력
    try:
        df = pd.read_csv(file_path_to_load, encoding='utf-8-sig', nrows=20000) # 샘플링해서 로드
        return clean_contour(df)
    except Exception as e:
        print(f"등고선 파일 로드 또는 정리 중 오류 발생: {e}")
        return None

def clean_contour(df):
    """등고선 데이터 컬럼 이름을 통일하고 유효 범위로 필터링."""
    try:
        height_col = x_col = y_col = None
        for col in df.columns:
            col_str = str(col).upper()
            if any(kw in col_str for kw in ['HEIGHT', '표고', '고도', 'ELEVATION', 'Z']): height_col = col # 'Z' 추가
            elif any(kw in col_str for kw in ['X', '경도', 'LON', 'LONGITUDE']): x_col = col
            elif any(kw in col_str for kw in ['Y', '위도', 'LAT', 'LATITUDE']): y_col = col
        
        if not all([height_col, x_col, y_col]):
            print("경고: 등고선 데이터에서 필수 컬럼 (표고, 경도, 위도)을 찾을 수 없어.")
            return None
        
        result = pd.DataFrame({
            '경도': pd.to_numeric(df[x_col], errors='coerce'),
            '위도': pd.to_numeric(df[y_col], errors='coerce'),
            '표고': pd.to_numeric(df[height_col], errors='coerce')
        }).dropna()
        
        # 서울 지역 대략적 범위 필터링
        seoul_filter = (
            (result['경도'] >= 125.0) & (result['경도'] <= 130.0) & # 경도 범위 조금 넓힘
            (result['위도'] >= 33.0) & (result['위도'] <= 39.0) &   # 위도 범위 조금 넓힘
            (result['표고'] >= -500) & (result['표고'] <= 5000)      # 표고 범위 넓힘
        )
        # 필터링 후 데이터가 남아있는지 확인
        if result[seoul_filter].empty:
            print("경고: 등고선 데이터가 서울 지역 범위 내에 없어. 필터링 후 데이터가 비어있어.")
            return None
        
        return result[seoul_filter]
    except Exception as e:
        print(f"등고선 데이터 정리 중 오류 발생: {e}")
        return None

def find_coord_columns(df):
    """데이터프레임에서 경도(X), 위도(Y) 컬럼을 찾아 반환."""
    x_col = y_col = None
    for col in df.columns:
        col_str = str(col).upper()
        if any(kw in col_str for kw in ['X', '경도', 'LON', 'LONGITUDE']): x_col = col
        elif any(kw in col_str for kw in ['Y', '위도', 'LAT', 'LATITUDE']): y_col = col
    return x_col, y_col

def find_target_column(df):
    """데이터프레임에서 타겟 변수 컬럼을 찾아 반환."""
    candidates = ['price', 'target', 'y', 'label', 'value'] 
    for candidate in candidates:
        for col in df.columns:
            if candidate.lower() in str(col).lower():
                return col
    # 그래도 못 찾으면, 숫자형 컬럼 중 마지막 컬럼을 반환
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return numeric_cols[-1] if len(numeric_cols) > 0 else None



