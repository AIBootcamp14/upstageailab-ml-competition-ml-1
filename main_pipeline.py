import os
import pandas as pd
from model.data_preprocessor import load_datasets, preprocess_and_feature_engineer
from model.model_trainer import evaluate_model_performance, train_and_predict_gradient_boosting, save_predictions
from model.utils import find_target_column

# --- 메인 파이프라인 실행 ---
if __name__ == "__main__":
    print("--- 부동산 가격 예측 파이프라인 시작 ---")

    # 1. 데이터 로드
    print("\n[단계 1/4] 데이터셋 로드 중...")
    DATA_DIR = 'data' # 데이터 파일들이 위치한 디렉토리
    datasets = load_datasets(data_dir=DATA_DIR)

    if datasets.get('train') is None or datasets.get('test') is None:
        print("오류: 훈련 또는 테스트 데이터셋을 로드할 수 없어 파이프라인을 중단합니다.")
    else:
        train_df = datasets['train']
        test_df = datasets['test']
        print(f"원본 훈련 데이터 크기: {train_df.shape}")
        print(f"원본 테스트 데이터 크기: {test_df.shape}")

        # 2. 기준 모델 성능 평가 (피처 엔지니어링 전)
        print("\n[단계 2/4] 피처 엔지니어링 전 기준 모델 성능 평가 중...")
        baseline_performance = evaluate_model_performance(train_df.copy(), model_type='baseline')
        if baseline_performance:
            print(f"기준 모델 (Gradient Boosting) RMSE: {baseline_performance['results']['Gradient Boosting']:.4f}")
            print(f"기준 모델 학습에 사용된 피처 수: {baseline_performance['feature_count']}")
        else:
            print("기준 모델 성능 평가에 실패했습니다.")

        # 3. 데이터 전처리 및 피처 엔지니어링
        print("\n[단계 3/4] 데이터 전처리 및 피처 엔지니어링 중...")
        enhanced_train_df = preprocess_and_feature_engineer(train_df.copy(), datasets, is_test=False)
        enhanced_test_df = preprocess_and_feature_engineer(test_df.copy(), datasets, is_test=True)

        if enhanced_train_df is None or enhanced_test_df is None:
            print("오류: 데이터 전처리 및 피처 엔지니어링에 실패하여 파이프라인을 중단합니다.")
        else:
            print(f"피처 엔지니어링 후 훈련 데이터 크기: {enhanced_train_df.shape}")
            print(f"피처 엔지니어링 후 테스트 데이터 크기: {enhanced_test_df.shape}")

            # 향상된 모델 성능 평가 (피처 엔지니어링 후)
            print("\n[단계 3.5/4] 피처 엔지니어링 후 향상된 모델 성능 평가 중...")
            enhanced_performance = evaluate_model_performance(enhanced_train_df.copy(), model_type='enhanced')
            if enhanced_performance:
                print(f"향상된 모델 (Gradient Boosting) RMSE: {enhanced_performance['results']['Gradient Boosting']:.4f}")
                print(f"향상된 모델 학습에 사용된 피처 수: {enhanced_performance['feature_count']}")
                print(f"새로 추가된 주요 피처 (일부): {', '.join(enhanced_performance['new_features'][:5])}...")
            else:
                print("향상된 모델 성능 평가에 실패했습니다.")

            # 4. 모델 훈련 및 예측
            print("\n[단계 4/4] 최종 모델 훈련 및 테스트 데이터 예측 중...")
            predictions_output = train_and_predict_gradient_boosting(enhanced_train_df, enhanced_test_df)

            if predictions_output:
                save_predictions(predictions_output, data_dir=DATA_DIR)
                print(f"\n--- 파이프라인 완료! 예측 결과가 '{DATA_DIR}' 폴더에 저장되었습니다. ---")
            else:
                print("오류: 모델 훈련 및 예측에 실패하여 결과를 저장할 수 없습니다.")
    
    print("\n--- 부동산 가격 예측 파이프라인 종료 ---")