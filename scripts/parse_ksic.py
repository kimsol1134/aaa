"""
한국표준산업분류(10차) 파싱 스크립트
"""
import pandas as pd
import numpy as np

def parse_ksic_classification(file_path):
    """
    한국표준산업분류 엑셀 파일을 파싱하여 대분류, 중분류 매핑 테이블 생성

    Parameters:
    -----------
    file_path : str
        산업분류 엑셀 파일 경로

    Returns:
    --------
    tuple : (대분류 DataFrame, 중분류 DataFrame, 전체 매핑 DataFrame)
    """
    df = pd.read_excel(file_path, header=None)

    # 대분류 추출 (컬럼 0-1, 3행부터)
    major_codes = []
    major_names = []

    for idx, row in df.iterrows():
        if idx < 3:
            continue

        code = row[0]
        name = row[1]

        if pd.notna(code) and isinstance(code, str) and len(code) == 1:
            major_codes.append(code)
            major_names.append(name)

    df_major = pd.DataFrame({
        '대분류코드': major_codes,
        '대분류명': major_names
    })

    # 중분류 추출 (컬럼 2-3, 3행부터)
    minor_codes = []
    minor_names = []
    current_major = None

    for idx, row in df.iterrows():
        if idx < 3:
            continue

        # 대분류 업데이트
        if pd.notna(row[0]) and isinstance(row[0], str) and len(row[0]) == 1:
            current_major = row[0]

        # 중분류 추출
        code = row[2]
        name = row[3]

        if pd.notna(code) and pd.notna(name):
            if isinstance(code, (int, float)):
                code = str(int(code)).zfill(2)
            elif isinstance(code, str):
                code = code.strip()

            if code and current_major:
                minor_codes.append(f"{current_major}{code}")
                minor_names.append(str(name).strip())

    df_minor = pd.DataFrame({
        '업종코드': minor_codes,
        '중분류명': minor_names
    })

    # 대분류 정보 병합
    df_minor['대분류코드'] = df_minor['업종코드'].str[0]
    df_mapping = df_minor.merge(df_major, on='대분류코드', how='left')

    return df_major, df_minor, df_mapping


if __name__ == '__main__':
    import os

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, 'data', '[붙임3-3] 한국표준산업분류(10차)_표.xlsx')
    output_path = os.path.join(base_dir, 'data', 'ksic_mapping.csv')

    df_major, df_minor, df_mapping = parse_ksic_classification(file_path)

    print("=== 대분류 ===")
    print(df_major)
    print(f"\n총 {len(df_major)}개")

    print("\n=== 중분류 (처음 20개) ===")
    print(df_minor.head(20))
    print(f"\n총 {len(df_minor)}개")

    print("\n=== 전체 매핑 테이블 (처음 20개) ===")
    print(df_mapping.head(20))

    # CSV 저장
    df_mapping.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 매핑 테이블 저장 완료: {output_path}")
