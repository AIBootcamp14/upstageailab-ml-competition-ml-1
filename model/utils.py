# 여러 모듈에서 공통적으로 사용할 유틸리티 함수



import pandas as pd
import numpy as np

def find_target_column(df):
    """
    데이터프레임에서 타겟 변수(예측하려는 값) 컬럼을 찾아 반환
    주요 후보 컬럼명들을 먼저 확인하고, 없으면 숫자형 컬럼 중 마지막 컬럼을 반환
    
    Args:
        df (pd.DataFrame): 타겟 컬럼을 찾을 데이터프레임.
        
    Returns:
        str or None: 찾은 타겟 컬럼의 이름. 없으면 None 반환.
    """
    candidates = ['price', 'target', 'y', 'label', 'value']
    
    for candidate in candidates:
        for col in df.columns:
            if candidate.lower() in str(col).lower():
                return col
    
    # 후보 컬럼명으로 못 찾으면, 숫자형 컬럼 중 마지막 컬럼을 타겟으로 가정
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return numeric_cols[-1] if len(numeric_cols) > 0 else None

# 이부분 나중에 다른 유틸리티 함수들이 추가될 수 있음