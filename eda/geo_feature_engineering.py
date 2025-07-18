# 지리적 피처 생성 함수들



# eda/geo_feature_engineering.py

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist # 거리 계산에 필요
import warnings
warnings.filterwarnings('ignore') # 경고 메시지 무시

# eda_utils 모듈에서 필요한 함수 import (find_coord_columns 함수만 사용)
from eda.eda_utils import find_coord_columns 

# --- 1. 개별 지리적 피처 생성 함수 ---

def add_academy_features(df, academy_df, x_col, y_col):
    """대형학원으로부터의 접근성 관련 피처를 추가."""
    try:
        # 학원 데이터의 경도/위도 컬럼 찾기 및 유효 데이터 필터링
        acad_x, acad_y = find_coord_columns(academy_df)
        if not acad_x or not acad_y or len(academy_df.dropna(subset=[acad_x, acad_y])) == 0: 
            print("경고: 유효한 학원 데이터 또는 컬럼 없어. 학원 피처 생성 건너뜀."); return df
        
        # 유효한 학원 좌표 추출
        coords = academy_df.dropna(subset=[acad_x, acad_y])[[acad_x, acad_y]].values
        features = []
        for idx in df.index:
            # 원본 DF의 좌표가 없으면 0으로 처리
            if pd.isna(df.loc[idx, x_col]) or pd.isna(df.loc[idx, y_col]): features.append([0, 0, 0]); continue
            point = np.array([[df.loc[idx, x_col], df.loc[idx, y_col]]])
            distances = cdist(point, coords)[0] # 현재 데이터 포인트와 모든 학원 간의 거리 계산
            features.append([
                (distances <= 0.005).sum(), # 500m 이내 학원 수 (0.005도는 대략 550m)
                (distances <= 0.01).sum(),  # 1km 이내 학원 수 (0.01도는 대략 1.1km)
                np.sum(1 / (distances + 0.001)) # 학원 접근성 (거리가 가까울수록 높은 값)
            ])
        # 생성된 피처를 DF에 추가
        feature_df = pd.DataFrame(features, columns=['academy_500m', 'academy_1km', 'academy_access'], index=df.index).fillna(0)
        df = pd.concat([df, feature_df], axis=1)
    except Exception as e: print(f"학원 피처 생성 중 오류 발생: {e}"); pass
    return df

def add_traffic_features(df, traffic_df, x_col, y_col):
    """통행불편지역 관련 피처를 추가."""
    try:
        traf_x, traf_y = find_coord_columns(traffic_df)
        if not traf_x or not traf_y or len(traffic_df.dropna(subset=[traf_x, traf_y])) == 0: 
            print("경고: 유효한 통행불편지역 데이터 또는 컬럼 없어. 통행 피처 생성 건너뜀."); return df
        
        coords = traffic_df.dropna(subset=[traf_x, traf_y])[[traf_x, traf_y]].values
        features = []
        for idx in df.index:
            if pd.isna(df.loc[idx, x_col]) or pd.isna(df.loc[idx, y_col]): features.append([0, 0]); continue
            point = np.array([[df.loc[idx, x_col], df.loc[idx, y_col]]])
            distances = cdist(point, coords)[0]
            features.append([
                np.sum(1 / (distances + 0.001)), # 통행불편지역 영향도 (가까울수록 영향 큼)
                (distances <= 0.01).sum() # 1km 반경 내 통행불편지역 수
            ])
        feature_df = pd.DataFrame(features, columns=['traffic_impact', 'traffic_count_1km'], index=df.index).fillna(0)
        df = pd.concat([df, feature_df], axis=1)
    except Exception as e: print(f"통행불편지역 피처 생성 중 오류 발생: {e}"); pass
    return df

def add_bus_features(df, bus_df, x_col, y_col):
    """버스정류장 관련 피처를 추가."""
    try:
        bus_x, bus_y = find_coord_columns(bus_df)
        if not bus_x or not bus_y or len(bus_df.dropna(subset=[bus_x, bus_y])) == 0: 
            print("경고: 유효한 버스정류장 데이터 또는 컬럼 없어. 버스 피처 생성 건너뜀."); return df
        
        valid_bus = bus_df.dropna(subset=[bus_x, bus_y])
        if len(valid_bus) > 5000: valid_bus = valid_bus.sample(n=5000, random_state=42) # 데이터가 너무 많으면 샘플링
        
        coords = valid_bus[[bus_x, bus_y]].values
        features = []
        for idx in df.index:
            if pd.isna(df.loc[idx, x_col]) or pd.isna(df.loc[idx, y_col]): features.append([0, 0, 0, 0]); continue
            point = np.array([[df.loc[idx, x_col], df.loc[idx, y_col]]])
            distances = cdist(point, coords)[0]
            features.append([
                distances.min(),             # 가장 가까운 버스정류장과의 거리
                (distances <= 0.002).sum(),  # 200m 이내 버스정류장 수 (0.002도는 대략 220m)
                (distances <= 0.005).sum(),  # 500m 이내 버스정류장 수
                np.sum(1 / (distances + 0.001)) # 버스정류장 접근성
            ])
        feature_df = pd.DataFrame(features, columns=['bus_min_dist', 'bus_200m', 'bus_500m', 'bus_access'], index=df.index).fillna(0)
        df = pd.concat([df, feature_df], axis=1)
    except Exception as e: print(f"버스정류장 피처 생성 중 오류 발생: {e}"); pass
    return df

def add_subway_features(df, subway_df, x_col, y_col):
    """지하철역 관련 피처를 추가."""
    try:
        sub_x, sub_y = find_coord_columns(subway_df)
        if not sub_x or not sub_y or len(subway_df.dropna(subset=[sub_x, sub_y])) == 0: 
            print("경고: 유효한 지하철역 데이터 또는 컬럼 없어. 지하철 피처 생성 건너뜀."); return df
        
        coords = subway_df.dropna(subset=[sub_x, sub_y])[[sub_x, sub_y]].values
        features = []
        for idx in df.index:
            if pd.isna(df.loc[idx, x_col]) or pd.isna(df.loc[idx, y_col]): features.append([0, 0, 0, 0]); continue
            point = np.array([[df.loc[idx, x_col], df.loc[idx, y_col]]])
            distances = cdist(point, coords)[0]
            features.append([
                distances.min(),             # 가장 가까운 지하철역과의 거리
                (distances <= 0.01).sum(),   # 1km 이내 지하철역 수
                (distances <= 0.02).sum(),   # 2km 이내 지하철역 수 (0.02도는 대략 2.2km)
                np.sum(1 / (distances + 0.001)) # 지하철역 접근성
            ])
        feature_df = pd.DataFrame(features, columns=['subway_min_dist', 'subway_1km', 'subway_2km', 'subway_access'], index=df.index).fillna(0)
        df = pd.concat([df, feature_df], axis=1)
    except Exception as e: print(f"지하철역 피처 생성 중 오류 발생: {e}"); pass
    return df

def add_contour_features(df, contour_df, x_col, y_col):
    """등고선(표고) 관련 피처를 추가 (표고 범위, 표준편차, 지형 유형)."""
    try:
        # contour_df의 필수 컬럼 존재 여부 및 유효 데이터 확인
        if not all(col in contour_df.columns for col in ['경도', '위도', '표고']) or len(contour_df.dropna(subset=['경도', '위도', '표고'])) == 0: 
            print("경고: 유효한 등고선 데이터 또는 컬럼 없어. 등고선 피처 생성 건너뜀."); return df
        
        valid_contour = contour_df.dropna(subset=['경도', '위도', '표고'])
        if len(valid_contour) > 20000: 
            valid_contour = valid_contour.sample(n=20000, random_state=42) # 데이터가 너무 많으면 샘플링
        
        coords = valid_contour[['경도', '위도']].values
        elevations = valid_contour['표고'].values
        features = []
        
        for idx in df.index:
            if pd.isna(df.loc[idx, x_col]) or pd.isna(df.loc[idx, y_col]): features.append([0, 0, 1]); continue # 기본값으로 0,0,1 추가
            point = np.array([[df.loc[idx, x_col], df.loc[idx, y_col]]])
            distances = cdist(point, coords)[0]
            
            # 가장 가까운 K개 등고선 지점의 표고 정보 사용
            k_nearest = min(8, len(distances))
            k_indices = np.argsort(distances)[:k_nearest]
            k_elevations = elevations[k_indices]
            k_distances = distances[k_indices]
            
            # 거리에 따른 가중치를 줘서 표고 추정
            # k_elevations가 비어있지 않은 경우에만 계산
            est_elevation = 0
            if len(k_elevations) > 0 and np.sum(1 / (k_distances + 1e-6)) > 0: # 가중치 합이 0이 아닌 경우에만
                 est_elevation = np.average(k_elevations, weights=1 / (k_distances + 1e-6))
            
            # 표고 범위 및 표준편차 계산
            if len(k_elevations) > 1:
                elev_range = k_elevations.max() - k_elevations.min()
                elev_std = k_elevations.std()
            else: # 등고선 지점 부족 시 0으로 처리
                elev_range = 0
                elev_std = 0
            
            # 추정된 표고에 따른 지형 유형 분류
            if est_elevation < 30: terrain = 1 # 저지대
            elif est_elevation < 70: terrain = 2 # 평지
            elif est_elevation < 120: terrain = 3 # 구릉지
            elif est_elevation < 200: terrain = 4 # 산비탈
            else: terrain = 5 # 고지대
            
            features.append([elev_range, elev_std, terrain])
        
        feature_df = pd.DataFrame(features, columns=['elevation_range', 'elevation_std', 'terrain_type'], index=df.index)
        # 결측치 처리: 모든 컬럼에 대해 평균으로 채우기 (데이터프레임이 비어있지 않을 경우)
        if not feature_df.empty:
            for col in feature_df.columns: 
                if feature_df[col].isnull().any(): # 해당 컬럼에 결측치가 있을 경우에만
                    feature_df[col] = feature_df[col].fillna(feature_df[col].mean()) 
                else: # 결측치가 없으면 그대로 유지
                    pass
        df = pd.concat([df, feature_df], axis=1)
    except Exception as e: print(f"등고선 피처 생성 중 오류 발생: {e}"); pass
    return df

def add_landmark_features(df, x_col, y_col):
    """서울 주요 랜드마크로부터의 거리를 피처로 추가."""
    try:
        landmarks = {
            'seoul_center': [126.9783, 37.5666], # 서울 시청
            'gangnam': [127.0276, 37.4979],      # 강남역
            'hongdae': [126.9235, 37.5563],      # 홍대입구역
            'jamsil': [127.1000, 37.5133],       # 잠실역 (롯데월드타워 근처)
            'yeouido': [126.9244, 37.5197],      # 여의도 (IFC몰 근처)
        }
        
        point_x = df[x_col].values
        point_y = df[y_col].values
        
        landmark_features = {}
        for landmark, coords in landmarks.items():
            distances = np.sqrt((point_x - coords[0])**2 + (point_y - coords[1])**2) # 유클리드 거리 계산
            landmark_features[f'{landmark}_dist'] = distances
        
        landmark_df = pd.DataFrame(landmark_features, index=df.index)
        # 결측치 처리: 모든 컬럼에 대해 평균으로 채우기 (데이터프레임이 비어있지 않을 경우)
        if not landmark_df.empty:
            for col in landmark_df.columns: 
                if landmark_df[col].isnull().any(): # 해당 컬럼에 결측치가 있을 경우에만
                    landmark_df[col] = landmark_df[col].fillna(landmark_df[col].mean()) 
                else: # 결측치가 없으면 그대로 유지
                    pass
        df = pd.concat([df, landmark_df], axis=1)
    except Exception as e: print(f"랜드마크 피처 생성 중 오류 발생: {e}"); pass
    return df

# --- 2. 모든 지리적 피처를 통합하여 생성하는 메인 함수 ---
def create_features_for_eda(base_df, academy_df, traffic_df, bus_df, subway_df, slope_df=None): # <-- 여기를 slope_df로 받도록 수정
    """EDA를 위해 여러 지리적 데이터셋으로부터 피처를 생성하고 통합."""
    # 기본 DF에서 경도/위도 컬럼 찾기
    x_col, y_col = find_coord_columns(base_df)
    if not x_col or not y_col:
        print("경고: 기본 데이터프레임에서 경도/위도 컬럼을 찾을 수 없어. 지리적 피처 생성 건너뜀.")
        return base_df # 피처 생성 없이 원본 DF 반환
    
    df = base_df.copy() # 원본 DF 손상 방지를 위해 복사본 사용
    
    print("새로운 지리적 피처 생성 중...")
    # 각 피처 생성 함수 호출 (데이터프레임이 None이 아닐 경우에만 실행)
    if academy_df is not None: df = add_academy_features(df, academy_df, x_col, y_col)
    if traffic_df is not None: df = add_traffic_features(df, traffic_df, x_col, y_col)
    if bus_df is not None: df = add_bus_features(df, bus_df, x_col, y_col)
    if subway_df is not None: df = add_subway_features(df, subway_df, x_col, y_col)
    
    # 등고선 데이터 처리: slope_df로 받아서 add_contour_features에 넘겨줌
    if slope_df is not None: df = add_contour_features(df, slope_df, x_col, y_col) 
    
    df = add_landmark_features(df, x_col, y_col) # 랜드마크는 외부 데이터 필요 없음
    
    print("모든 지리적 피처 생성 완료.")
    return df

# 이 파일은 함수들을 모아두는 용도이므로, 직접 실행할 코드는 포함하지 않음.








