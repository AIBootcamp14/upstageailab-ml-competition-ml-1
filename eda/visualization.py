# eda/visualization.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings
import os 
warnings.filterwarnings('ignore') 

# EDA 프로세스의 핵심 함수: 피처들을 분석하고 시각화 결과를 저장
def perform_eda_on_features(df, target_col):
    print("\n--- 생성된 피처 EDA 시작 ---")
    
    # 결과 저장 폴더 경로 설정 및 생성
    # 현재 작업 디렉토리 (project_root) 기준으로 'eda/results' 폴더를 찾아 생성
    results_dir = os.path.join(os.getcwd(), 'eda', 'results') 
    os.makedirs(results_dir, exist_ok=True) 
    print(f"결과가 저장될 디렉토리: {results_dir}") 
    
    # 숫자형 피처만 선택하여 분석
    numeric_df = df.select_dtypes(include=np.number)
    print(f"총 {len(numeric_df.columns)}개의 숫자형 피처 분석.")

    # 1. 통계 요약 출력
    print("\n--- 새로운 피처 통계 요약 ---")
    # describe() 결과를 문자열로 변환하여 파일에 저장하기 위해
    summary_stats = numeric_df.describe().to_string()
    print(summary_stats) # 터미널에도 출력

    # 2. 타겟 변수와의 상관관계 분석
    correlations = pd.Series(dtype=float) # 초기화
    if target_col in numeric_df.columns and not numeric_df[target_col].isnull().all():
        print("\n--- 타겟 변수 'target'와의 상관관계 (새로운 피처) ---")
        # target 컬럼과 다른 모든 숫자형 피처 간의 상관관계 계산
        # target_col 자신과의 상관관계는 1.0이므로 시각화에서 제외하기 위해 나중에 처리
        correlations = numeric_df.corr()[target_col].sort_values(ascending=False)
        print(correlations.to_string()) # 터미널에도 출력
    else:
        print(f"경고: 타겟 컬럼 '{target_col}'을(를) 찾을 수 없거나 모든 값이 NaN이야. 상관관계를 계산하지 않을게.")
        
    # 3. 상관관계가 높은 상위 N개 피처 분포 시각화 및 저장
    top_n = 5 # 시각화할 상위 N개 피처
    
    # 상관관계가 비어있지 않고, 타겟 컬럼 외에 시각화할 피처가 있는 경우에만 시각화 진행
    if not correlations.empty and len(correlations) > 1: # 최소한 target_col 외에 다른 피처가 있어야 함
        # target_col 자신을 제외하고, 상관관계 상위 top_n개 피처 선택
        # dropna()를 통해 NaN 상관관계를 가진 피처는 제외 (예: academy_500m)
        features_to_plot = [f for f in correlations.index if f != target_col and not pd.isna(correlations[f])][:top_n]
        
        if features_to_plot: # 실제로 시각화할 피처가 있는 경우에만 진행
            print("\n--- 타겟 변수와 상관관계 높은 상위 5개 피처 분포 시각화 ---")

            # 히스토그램 시각화
            plt.figure(figsize=(15, 10))
            for i, feature in enumerate(features_to_plot): 
                plt.subplot(2, 3, i + 1) # 2행 3열 서브플롯
                sns.histplot(numeric_df[feature].dropna(), kde=True, bins=30) 
                plt.title(f'{feature} 분포', fontsize=12) 
                plt.xlabel(feature, fontsize=10)
                plt.ylabel('개수', fontsize=10) 
            plt.tight_layout() 
            hist_plot_filepath = os.path.join(results_dir, 'hist_top_geo_features.png')
            print(f"히스토그램 저장 시도: {hist_plot_filepath}")
            try:
                plt.savefig(hist_plot_filepath)
                print("히스토그램 저장 성공!")
            except Exception as e:
                print(f"히스토그램 저장 실패: {e}")
            plt.close() # 중요: 서버 환경에서 창 닫기

            # 산점도 시각화
            plt.figure(figsize=(15, 10))
            for i, feature in enumerate(features_to_plot):
                plt.subplot(2, 3, i + 1) # 2행 3열 서브플롯
                sns.scatterplot(x=numeric_df[feature], y=numeric_df[target_col], alpha=0.6) 
                plt.title(f'{feature} vs {target_col}', fontsize=12)
                plt.xlabel(feature, fontsize=10)
                plt.ylabel(target_col, fontsize=10)
            plt.tight_layout()
            scatter_plot_filepath = os.path.join(results_dir, 'scatter_top_geo_features_vs_target.png')
            print(f"산점도 저장 시도: {scatter_plot_filepath}") 
            try:
                plt.savefig(scatter_plot_filepath)
                print("산점도 저장 성공!")
            except Exception as e:
                print(f"산점도 저장 실패: {e}")
            plt.close() # 중요: 서버 환경에서 창 닫기
        else:
            print("시각화할 유의미한 피처가 없어서 시각화를 건너뛸게.")
    else:
        print("상관관계 분석을 위한 데이터가 없거나, 타겟 컬럼 외 피처가 부족해서 시각화를 건너뛸게.")

    # EDA 결과 요약 텍스트 파일 저장
    summary_filepath = os.path.join(results_dir, 'eda_summary.txt')
    print(f"EDA 요약 결과를 '{summary_filepath}'에 저장 시도...")
    try:
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            f.write("--- 새로운 피처 통계 요약 ---\n")
            f.write(summary_stats)
            f.write("\n\n--- 타겟 변수 'target'와의 상관관계 (새로운 피처) ---\n")
            if not correlations.empty:
                f.write(correlations.to_string())
            else:
                f.write("상관관계 데이터 없음.\n")
        print("EDA 요약 결과 저장 성공!")
    except Exception as e:
        print(f"EDA 요약 결과 저장 실패: {e}")


    print("\n--- 생성된 피처 EDA 완료 ---")
    print(f"EDA 결과가 '{results_dir}' 폴더에 저장되었어.")