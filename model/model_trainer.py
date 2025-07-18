# 모델 학습,평가,예측 관련된 기능 모음


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import os
import warnings
warnings.filterwarnings('ignore')

# utils 모듈에서 find_target_column 함수를 가져와 사용
from model.utils import find_target_column

def evaluate_model_performance(df, model_type='baseline'):
    """
    모델 성능 평가 (기존 또는 향상된 데이터셋에 대해 Gradient Boosting 모델만 사용).
    
    Args:
        df (pd.DataFrame): 평가할 데이터프레임 (train_df 또는 enhanced_df).
        model_type (str): 'baseline' 또는 'enhanced'를 지정하여 출력 메시지 구분.
        
    Returns:
        dict: RMSE 결과, 피처 수, 새로운 피처 목록 (enhanced일 경우).
    """
    try:
        target_col = find_target_column(df)
        if not target_col:
            print(f"경고: {model_type} 모델 평가를 위한 타겟 컬럼을 찾을 수 없어.")
            return None
        
        feature_cols = [col for col in df.columns if col != target_col]
        numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns
        
        # 결측값은 평균으로 채우기
        X = df[numeric_features].fillna(df[numeric_features].mean())
        y = df[target_col].fillna(df[target_col].mean())
        
        if len(X) < 2 or len(y) < 2:
            print(f"경고: {model_type} 모델 평가를 위한 데이터 포인트가 너무 적어.")
            return None

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model_name = 'Gradient Boosting'
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        print(f" - {model_type} {model_name} 모델 학습 및 평가 중...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results = {model_name: rmse}
        print(f"   - {model_type} {model_name} RMSE: {rmse:.4f}")
        
        # 향상된 모델일 경우에만 새로운 피처 목록 반환
        new_features = []
        if model_type == 'enhanced':
            new_keywords = ['academy', 'traffic', 'bus', 'subway', 'elevation', 'terrain', 
                            'seoul_center', 'gangnam', 'hongdae', 'jamsil', 'yeouido', # 랜드마크 키워드 추가
                            'dist', 'access', 'count', 'impact', 'range', 'std']
            new_features = [col for col in numeric_features if any(kw in col for kw in new_keywords)]
            
        return {'results': results, 'feature_count': len(numeric_features), 'new_features': new_features}
        
    except Exception as e:
        print(f"{model_type} 모델 평가 중 오류 발생: {e}")
        return None

def train_and_predict_gradient_boosting(enhanced_train_df, enhanced_test_df):
    """
    Gradient Boosting 모델을 훈련하고 테스트 데이터를 예측하는 함수.
    
    Args:
        enhanced_train_df (pd.DataFrame): 피처 엔지니어링이 완료된 훈련 데이터.
        enhanced_test_df (pd.DataFrame): 피처 엔지니어링이 완료된 테스트 데이터.
        
    Returns:
        dict: 모델 이름, 예측 결과 (numpy array), 교차 검증 평균 점수.
    """
    try:
        target_col = find_target_column(enhanced_train_df)
        if not target_col:
            print("단일 모델 학습을 위한 타겟 컬럼을 찾을 수 없어.")
            return None
        
        feature_cols = [col for col in enhanced_train_df.columns if col != target_col]
        numeric_features = enhanced_train_df[feature_cols].select_dtypes(include=[np.number]).columns
        
        X_train = enhanced_train_df[numeric_features].fillna(enhanced_train_df[numeric_features].mean())
        y_train = enhanced_train_df[target_col].fillna(enhanced_train_df[target_col].mean())
        
        if enhanced_test_df is not None:
            # 테스트 데이터의 피처가 훈련 데이터와 일치하도록 보장
            # 훈련 데이터에 없는 피처는 제거, 테스트 데이터에 없는 피처는 0으로 채움 (선택사항)
            # 여기서는 훈련 데이터에 있는 피처만 사용하도록 통일
            test_features_aligned = [col for col in numeric_features if col in enhanced_test_df.columns]
            X_test = enhanced_test_df[test_features_aligned].fillna(enhanced_test_df[test_features_aligned].mean())
            X_train = X_train[test_features_aligned] # 훈련 데이터도 동일한 피처 셋 사용
        else:
            print("테스트 데이터가 없어 단일 모델 예측을 수행할 수 없어.")
            return None
        
        model_name = 'Gradient Boosting'
        # 여기서는 하이퍼파라미터를 200, max_depth=8로 고정했어. 필요하면 조정해.
        model = GradientBoostingRegressor(n_estimators=200, max_depth=8, random_state=42) 
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_scores = []
        
        print(f" - {model_name} K-Fold 교차 검증 중...")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model.fit(X_fold_train, y_fold_train)
            y_pred_val = model.predict(X_fold_val)
            
            rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred_val))
            fold_scores.append(rmse)
        
        avg_score = np.mean(fold_scores)
        print(f"   - {model_name} 평균 RMSE: {avg_score:.4f}")
        
        # 전체 훈련 데이터로 최종 모델 학습 및 테스트 데이터 예측
        print(f" - 전체 훈련 데이터로 {model_name} 모델 최종 학습 중...")
        model.fit(X_train, y_train)
        test_pred = model.predict(X_test)
        
        print(f"\n{model_name} 예측 완료.")
        return {
            'model_name': model_name,
            'predictions': test_pred,
            'score': avg_score
        }
        
    except Exception as e:
        print(f"단일 모델 학습 중 오류 발생: {e}")
        return None

def save_predictions(test_predictions_output, data_dir='data'):
    """
    예측 결과를 CSV 파일로 저장하는 함수.
    
    Args:
        test_predictions_output (dict): train_and_predict_gradient_boosting 함수의 반환값.
        data_dir (str): 결과를 저장할 디렉토리 경로.
    """
    try:
        # data_dir이 상대 경로일 경우 현재 작업 디렉토리를 기준으로 절대 경로를 얻음
        # 이 함수가 호출될 때 이미 작업 디렉토리가 적절히 설정되어 있다고 가정
        output_dir = os.path.abspath(data_dir)
        os.makedirs(output_dir, exist_ok=True) # 디렉토리가 없으면 생성

        if test_predictions_output and isinstance(test_predictions_output, dict) and 'predictions' in test_predictions_output:
            model_name = test_predictions_output['model_name'].replace(' ', '_').lower()
            pred = test_predictions_output['predictions']
            
            # 예측값을 반올림하여 정수 타입으로 변환
            final_predictions = np.round(pred).astype(int)
            
            submission_df = pd.DataFrame({
                'id': range(len(final_predictions)),
                'target': final_predictions
            })
            
            filepath = os.path.join(output_dir, f'submission_{model_name}_rounded_int.csv')
            submission_df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"단일 모델 예측 결과 '{filepath}' 저장 완료 (정수 타입).")
            print("총 21개의 지리적 피처가 모델 학습에 사용되었음을 가정합니다.")
        else:
            print("저장할 예측 결과가 없어.")
        
    except Exception as e:
        print(f"결과 저장 중 오류 발생: {e}")

# 이 모듈이 직접 실행될 때는 테스트 목적으로만 사용 (메인 파이프라인에서 호출 예정)
if __name__ == "__main__":
    print("이 스크립트는 주로 다른 메인 파이프라인 스크립트에서 호출됩니다.")
    print("테스트를 위해 가상 데이터를 생성하여 실행합니다.")

    # 가상 데이터 생성 (실제 사용 시에는 피처 엔지니어링된 실제 데이터로 교체해야 함)
    dummy_train = pd.DataFrame({
        '경도': np.random.rand(100) * 0.1 + 127,
        '위도': np.random.rand(100) * 0.1 + 37.5,
        'feature1': np.random.rand(100) * 100,
        'feature2': np.random.rand(100) * 50,
        'target': np.random.rand(100) * 1000 + 500
    })
    
    # 더미 데이터에 추가 피처 (예: academy_500m, traffic_impact 등) 추가
    dummy_train['academy_500m'] = np.random.randint(0, 5, 100)
    dummy_train['traffic_impact'] = np.random.rand(100) * 10
    dummy_train['bus_min_dist'] = np.random.rand(100) * 0.01
    dummy_train['subway_1km'] = np.random.randint(0, 3, 100)
    dummy_train['elevation_range'] = np.random.rand(100) * 200
    dummy_train['terrain_type'] = np.random.randint(1, 6, 100)
    dummy_train['seoul_center_dist'] = np.random.rand(100) * 0.5


    dummy_test = pd.DataFrame({
        'id': range(20), # ID 컬럼 추가
        '경도': np.random.rand(20) * 0.1 + 127,
        '위도': np.random.rand(20) * 0.1 + 37.5,
        'feature1': np.random.rand(20) * 100,
        'feature2': np.random.rand(20) * 50,
        # 테스트 데이터는 타겟 컬럼이 없어야 함
    })

    # 더미 테스트 데이터에도 동일한 추가 피처 추가
    dummy_test['academy_500m'] = np.random.randint(0, 5, 20)
    dummy_test['traffic_impact'] = np.random.rand(20) * 10
    dummy_test['bus_min_dist'] = np.random.rand(20) * 0.01
    dummy_test['subway_1km'] = np.random.randint(0, 3, 20)
    dummy_test['elevation_range'] = np.random.rand(20) * 200
    dummy_test['terrain_type'] = np.random.randint(1, 6, 20)
    dummy_test['seoul_center_dist'] = np.random.rand(20) * 0.5


    print("\n--- 더미 훈련 데이터로 모델 성능 평가 (기존) ---")
    baseline_perf = evaluate_model_performance(dummy_train, model_type='baseline')
    print(baseline_perf)

    print("\n--- 더미 훈련 데이터로 모델 성능 평가 (향상) ---")
    enhanced_perf = evaluate_model_performance(dummy_train, model_type='enhanced')
    print(enhanced_perf)

    print("\n--- 더미 데이터로 최종 모델 학습 및 예측 ---")
    predictions_output = train_and_predict_gradient_boosting(dummy_train, dummy_test)
    if predictions_output:
        print(f"예측 결과 샘플: {predictions_output['predictions'][:5]}")
        # 임시 디렉토리에 저장 (실제 사용 시에는 DATA_DIR을 전달해야 함)
        save_predictions(predictions_output, data_dir='temp_results')