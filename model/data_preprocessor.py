# 데이터 전처리와 피처 엔지니어링 관련 함수 모음

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import cKDTree
from geopy.distance import geodesic
import warnings
warnings.filterwarnings('ignore')

# utils 모듈에서 find_target_column 함수를 가져와 사용
from model.utils import find_target_column

def load_csv(filepath):
    """CSV 파일을 로드하고, 파일이 없으면 경고 메시지를 출력해."""
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        print(f"경고: 파일이 없어 - {filepath}")
        return None

def load_datasets(data_dir='data'):
    """
    필요한 모든 데이터셋을 로드해.
    
    Args:
        data_dir (str): 데이터 파일들이 있는 디렉토리 경로.
        
    Returns:
        dict: 로드된 데이터프레임들을 담은 딕셔너리.
    """
    datasets = {}
    
    # os.path.join을 사용하여 경로를 안전하게 구성
    base_path = os.path.abspath(data_dir) # 절대 경로로 변환하여 안정성 확보
    
    datasets['train'] = load_csv(os.path.join(base_path, 'concat_select_csv2-5.csv'))
    datasets['test'] = load_csv(os.path.join(base_path, 'test2.csv'))
    datasets['academy'] = load_csv(os.path.join(base_path, 'processed_academy.csv'))
    datasets['traffic'] = load_csv(os.path.join(base_path, 'traffic_data.csv'))
    datasets['bus'] = load_csv(os.path.join(base_path, 'bus_station_data.csv'))
    datasets['subway'] = load_csv(os.path.join(base_path, 'subway_station_data.csv'))
    datasets['contour'] = load_contour_data(base_path) # 등고선 데이터는 별도 함수로 로드
    
    # 데이터 로드 확인 및 경고 메시지
    for name, df in datasets.items():
        if df is None:
            print(f"주의: '{name}' 데이터셋 로드 실패.")
        else:
            print(f"'{name}' 데이터셋 로드 완료. ({len(df)} 행)")
            
    return datasets

def load_contour_data(data_dir):
    """등고선 데이터를 로드하고 전처리해."""
    contour_filepath = os.path.join(data_dir, 'contour.csv')
    df_contour = load_csv(contour_filepath)
    if df_contour is not None:
        df_contour = clean_contour(df_contour)
        print(f"'contour' 데이터셋 전처리 완료. ({len(df_contour)} 행)")
    return df_contour

def clean_contour(df):
    """등고선 데이터프레임의 컬럼명을 정리해."""
    if df is not None:
        df.columns = ['id', '경도', '위도', '등고선']
    return df

def find_coord_columns(df):
    """데이터프레임에서 경도, 위도 컬럼을 찾아 반환해."""
    lon_col = None
    lat_col = None
    
    for col in df.columns:
        if '경도' in col or 'lon' in col.lower() or 'longitude' in col.lower():
            lon_col = col
        if '위도' in col or 'lat' in col.lower() or 'latitude' in col.lower():
            lat_col = col
            
    if lon_col is None or lat_col is None:
        raise ValueError("경도 또는 위도 컬럼을 찾을 수 없어.")
        
    return lon_col, lat_col

def add_spatial_features(base_df, target_df, target_name, radius_km, feature_name_prefix, lon_col_base, lat_col_base, lon_col_target, lat_col_target):
    """
    특정 반경 내의 지리적 피처를 추가해.
    
    Args:
        base_df (pd.DataFrame): 기준이 되는 데이터프레임 (예: 훈련/테스트 데이터).
        target_df (pd.DataFrame): 피처를 가져올 대상 데이터프레임 (예: 학원, 버스 정류장 등).
        target_name (str): 대상 데이터의 이름 (예: 'academy', 'bus').
        radius_km (float): 반경 (킬로미터 단위).
        feature_name_prefix (str): 생성될 피처 이름의 접두사.
        lon_col_base (str): base_df의 경도 컬럼명.
        lat_col_base (str): base_df의 위도 컬럼명.
        lon_col_target (str): target_df의 경도 컬럼명.
        lat_col_target (str): target_df의 위도 컬럼명.
        
    Returns:
        pd.DataFrame: 피처가 추가된 base_df.
    """
    if target_df is None or base_df is None:
        print(f"경고: {target_name} 또는 기준 데이터프레임이 없어 공간 피처를 추가할 수 없어.")
        return base_df

    base_coords = base_df[[lon_col_base, lat_col_base]].values
    target_coords = target_df[[lon_col_target, lat_col_target]].values

    # cKDTree를 사용하여 가장 가까운 이웃 검색
    tree = cKDTree(target_coords)
    
    # 각 기준점으로부터 반경 내의 모든 대상점 찾기
    indices_in_radius = tree.query_ball_point(base_coords, r=radius_km / 111.32) # 대략적인 위도/경도 -> km 변환

    # 반경 내 개수 피처
    base_df[f'{feature_name_prefix}_count_{int(radius_km*1000)}m'] = [len(idx) for idx in indices_in_radius]

    # 가장 가까운 거리 피처
    distances, _ = tree.query(base_coords, k=1)
    base_df[f'{feature_name_prefix}_min_dist_km'] = distances * 111.32 # 다시 km로 변환
    
    return base_df

def add_academy_features(df, academy_df):
    """학원 관련 피처를 추가해."""
    lon_col_df, lat_col_df = find_coord_columns(df)
    lon_col_academy, lat_col_academy = find_coord_columns(academy_df)
    
    # 500m 반경 내 학원 수
    df = add_spatial_features(df, academy_df, 'academy', 0.5, 'academy', lon_col_df, lat_col_df, lon_col_academy, lat_col_academy)
    # 가장 가까운 학원까지의 거리
    # add_spatial_features 함수 내에서 자동으로 min_dist_km가 추가됨
    return df

def add_traffic_features(df, traffic_df):
    """교통량 관련 피처를 추가해."""
    if traffic_df is None or df is None:
        print("경고: 교통량 데이터 또는 기준 데이터프레임이 없어 교통량 피처를 추가할 수 없어.")
        return df

    lon_col_df, lat_col_df = find_coord_columns(df)
    lon_col_traffic, lat_col_traffic = find_coord_columns(traffic_df)

    # 교통량 데이터의 위도, 경도 컬럼 이름이 다를 수 있으므로 확인
    if '경도' not in traffic_df.columns or '위도' not in traffic_df.columns:
        print("경고: 교통량 데이터에 '경도' 또는 '위도' 컬럼이 없어. 피처 추가 불가.")
        return df

    # 가장 가까운 교통량 측정 지점의 교통량 정보 추가
    traffic_coords = traffic_df[[lon_col_traffic, lat_col_traffic]].values
    base_coords = df[[lon_col_df, lat_col_df]].values

    tree = cKDTree(traffic_coords)
    distances, indices = tree.query(base_coords, k=1) # 가장 가까운 1개 지점

    # 가장 가까운 지점의 '교통량' 컬럼 값 가져오기
    # '교통량' 컬럼이 traffic_df에 있는지 확인
    if '교통량' in traffic_df.columns:
        df['closest_traffic_volume'] = traffic_df['교통량'].iloc[indices].values
    else:
        print("경고: 교통량 데이터에 '교통량' 컬럼이 없어. 'closest_traffic_volume' 피처 추가 불가.")
        df['closest_traffic_volume'] = np.nan # 컬럼이 없으면 NaN으로 채움

    # 필요한 경우 교통량 관련 다른 통계 피처 추가 (예: 반경 내 평균 교통량 등)
    # 여기서는 가장 가까운 지점의 교통량만 추가했어.
    
    return df

def add_bus_features(df, bus_df):
    """버스 정류장 관련 피처를 추가해."""
    lon_col_df, lat_col_df = find_coord_columns(df)
    lon_col_bus, lat_col_bus = find_coord_columns(bus_df)
    
    # 500m 반경 내 버스 정류장 수
    df = add_spatial_features(df, bus_df, 'bus', 0.5, 'bus', lon_col_df, lat_col_df, lon_col_bus, lat_col_bus)
    # 가장 가까운 버스 정류장까지의 거리
    # add_spatial_features 함수 내에서 자동으로 min_dist_km가 추가됨
    return df

def add_subway_features(df, subway_df):
    """지하철역 관련 피처를 추가해."""
    lon_col_df, lat_col_df = find_coord_columns(df)
    lon_col_subway, lat_col_subway = find_coord_columns(subway_df)
    
    # 1km 반경 내 지하철역 수
    df = add_spatial_features(df, subway_df, 'subway', 1.0, 'subway', lon_col_df, lat_col_df, lon_col_subway, lat_col_subway)
    # 가장 가까운 지하철역까지의 거리
    # add_spatial_features 함수 내에서 자동으로 min_dist_km가 추가됨
    return df

def add_contour_features(df, contour_df):
    """등고선(고도) 관련 피처를 추가해."""
    if contour_df is None or df is None:
        print("경고: 등고선 데이터 또는 기준 데이터프레임이 없어 등고선 피처를 추가할 수 없어.")
        return df

    lon_col_df, lat_col_df = find_coord_columns(df)
    lon_col_contour, lat_col_contour = find_coord_columns(contour_df)

    # 등고선 데이터의 위도, 경도 컬럼 이름이 다를 수 있으므로 확인
    if '등고선' not in contour_df.columns:
        print("경고: 등고선 데이터에 '등고선' 컬럼이 없어. 피처 추가 불가.")
        return df

    # 가장 가까운 등고선 지점의 고도 정보 추가
    contour_coords = contour_df[[lon_col_contour, lat_col_contour]].values
    base_coords = df[[lon_col_df, lat_col_df]].values

    tree = cKDTree(contour_coords)
    distances, indices = tree.query(base_coords, k=1) # 가장 가까운 1개 지점

    df['closest_elevation'] = contour_df['등고선'].iloc[indices].values
    
    # 추가적으로 고도 변화량, 고도 범위 등 더 복잡한 피처를 만들 수 있어.
    # 예: 주변 5개 지점의 고도 표준편차 등
    
    return df

def add_landmark_features(df):
    """
    주요 랜드마크(서울 중심, 강남, 홍대, 잠실, 여의도)까지의 거리를 피처로 추가해.
    
    Args:
        df (pd.DataFrame): 피처를 추가할 데이터프레임.
        
    Returns:
        pd.DataFrame: 랜드마크 거리 피처가 추가된 데이터프레임.
    """
    lon_col, lat_col = find_coord_columns(df)
    
    landmarks = {
        'seoul_center': (37.5665, 126.9780),  # 서울 시청 기준
        'gangnam': (37.5172, 127.0473),       # 강남역 기준
        'hongdae': (37.5577, 126.9246),       # 홍대입구역 기준
        'jamsil': (37.5145, 127.1059),        # 잠실역 기준
        'yeouido': (37.5218, 126.9242)        # 여의도역 기준
    }
    
    for name, coords in landmarks.items():
        df[f'dist_from_{name}_km'] = df.apply(
            lambda row: geodesic((row[lat_col], row[lon_col]), coords).km, axis=1
        )
    return df

def preprocess_and_feature_engineer(df, datasets, is_test=False):
    """
    데이터 전처리 및 피처 엔지니어링을 수행하는 메인 함수.
    
    Args:
        df (pd.DataFrame): 전처리할 기본 데이터프레임 (train 또는 test).
        datasets (dict): 로드된 모든 데이터셋 (academy, traffic 등).
        is_test (bool): 테스트 데이터인 경우 True (타겟 컬럼 처리 등 구분).
        
    Returns:
        pd.DataFrame: 피처 엔지니어링이 완료된 데이터프레임.
    """
    if df is None:
        print("경고: 입력 데이터프레임이 없어 전처리 및 피처 엔지니어링을 수행할 수 없어.")
        return None

    processed_df = df.copy()

    # 1. 컬럼명 정리 및 불필요 컬럼 제거 (필요하다면 추가)
    # 예: 'Unnamed: 0' 컬럼 제거
    if 'Unnamed: 0' in processed_df.columns:
        processed_df = processed_df.drop(columns=['Unnamed: 0'])
    
    # 2. 결측치 처리 (여기서는 일단 평균으로 채우는 예시, 더 정교한 방법 사용 가능)
    # 숫자형 컬럼에 대해서만 적용
    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if processed_df[col].isnull().any():
            processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
            
    # 3. 범주형 변수 인코딩 (필요하다면 추가)
    # 예: one-hot encoding for '지역' or '건물유형' if they exist and are categorical
    # processed_df = pd.get_dummies(processed_df, columns=['지역'], dummy_na=False)

    # 4. 지리적 피처 엔지니어링
    print(" - 지리적 피처 엔지니어링 중...")
    processed_df = add_academy_features(processed_df, datasets.get('academy'))
    processed_df = add_traffic_features(processed_df, datasets.get('traffic'))
    processed_df = add_bus_features(processed_df, datasets.get('bus'))
    processed_df = add_subway_features(processed_df, datasets.get('subway'))
    processed_df = add_contour_features(processed_df, datasets.get('contour'))
    processed_df = add_landmark_features(processed_df) # 랜드마크 거리 피처 추가

    # 5. 스케일링 (선택 사항, 모델에 따라 필요)
    # 여기서는 Gradient Boosting이라 필수는 아니지만, 다른 모델을 위해 스케일링을 추가할 수 있어.
    # scaler = MinMaxScaler()
    # for col in numeric_cols: # 타겟 컬럼은 스케일링하지 않음
    #     if col != find_target_column(processed_df):
    #         processed_df[col] = scaler.fit_transform(processed_df[[col]])

    print(f" - 피처 엔지니어링 완료. 현재 피처 수: {len(processed_df.columns)}")
    return processed_df

# 이 모듈이 직접 실행될 때는 테스트 목적으로만 사용 (메인 파이프라인에서 호출 예정)
if __name__ == "__main__":
    print("이 스크립트는 주로 다른 메인 파이프라인 스크립트에서 호출됩니다.")
    print("테스트를 위해 가상 데이터를 생성하여 피처 엔지니어링을 실행합니다.")

    # 가상 데이터 생성 (실제 사용 시에는 실제 데이터로 교체해야 함)
    dummy_train_base = pd.DataFrame({
        'id': range(100),
        '경도': np.random.rand(100) * 0.1 + 127,
        '위도': np.random.rand(100) * 0.1 + 37.5,
        'feature_A': np.random.rand(100) * 100,
        'target': np.random.rand(100) * 1000 + 500
    })
    
    dummy_test_base = pd.DataFrame({
        'id': range(20),
        '경도': np.random.rand(20) * 0.1 + 127,
        '위도': np.random.rand(20) * 0.1 + 37.5,
        'feature_A': np.random.rand(20) * 100,
    })

    dummy_academy = pd.DataFrame({
        '경도': np.random.rand(50) * 0.1 + 127,
        '위도': np.random.rand(50) * 0.1 + 37.5,
    })
    dummy_traffic = pd.DataFrame({
        '경도': np.random.rand(30) * 0.1 + 127,
        '위도': np.random.rand(30) * 0.1 + 37.5,
        '교통량': np.random.randint(1000, 10000, 30)
    })
    dummy_bus = pd.DataFrame({
        '경도': np.random.rand(70) * 0.1 + 127,
        '위도': np.random.rand(70) * 0.1 + 37.5,
    })
    dummy_subway = pd.DataFrame({
        '경도': np.random.rand(20) * 0.1 + 127,
        '위도': np.random.rand(20) * 0.1 + 37.5,
    })
    dummy_contour = pd.DataFrame({
        '경도': np.random.rand(100) * 0.1 + 127,
        '위도': np.random.rand(100) * 0.1 + 37.5,
        '등고선': np.random.rand(100) * 500
    })

    dummy_datasets = {
        'train': dummy_train_base,
        'test': dummy_test_base,
        'academy': dummy_academy,
        'traffic': dummy_traffic,
        'bus': dummy_bus,
        'subway': dummy_subway,
        'contour': dummy_contour
    }

    print("\n--- 훈련 데이터 피처 엔지니어링 테스트 ---")
    enhanced_train_df = preprocess_and_feature_engineer(dummy_train_base, dummy_datasets, is_test=False)
    if enhanced_train_df is not None:
        print(f"훈련 데이터 피처 엔지니어링 완료. 최종 컬럼 수: {len(enhanced_train_df.columns)}")
        print(enhanced_train_df.head())

    print("\n--- 테스트 데이터 피처 엔지니어링 테스트 ---")
    enhanced_test_df = preprocess_and_feature_engineer(dummy_test_base, dummy_datasets, is_test=True)
    if enhanced_test_df is not None:
        print(f"테스트 데이터 피처 엔지니어링 완료. 최종 컬럼 수: {len(enhanced_test_df.columns)}")
        print(enhanced_test_df.head())



