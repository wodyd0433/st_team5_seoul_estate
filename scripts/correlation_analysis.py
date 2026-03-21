import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib
from scipy.stats import pearsonr
import os

# 디렉토리 설정
BASE_DIR = r"c:\Users\wodyd\OneDrive\PythonWorkspace\icb7\korea_estate"
DATASETS_RAW = os.path.join(BASE_DIR, "datasets", "raw")
DATASETS_CLEANED = os.path.join(BASE_DIR, "datasets", "cleaned", "housing")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
REPORT_DIR = os.path.join(BASE_DIR, "report")

# 필요한 디렉토리 생성
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

def load_data():
    # 1. 전세 사기 빈도 데이터 로드
    scam_path = os.path.join(DATASETS_RAW, "rental-scam-frequency.csv")
    
    # 서울시 25개 자치구 표준 순서 (행정구역 코드 순)
    seoul_gus = [
        "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구", 
        "강북구", "도봉구", "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구", 
        "구로구", "금천구", "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구", "강동구"
    ]
    
    # 인코딩 무관하게 데이터 값만 추출 (corrupted names 무시)
    scam_df = pd.read_csv(scam_path, encoding='latin-1', header=None, skiprows=1)
    
    # 25개 구 데이터만 추출 (Header 제외 1~25행)
    scam_data = scam_df.iloc[:25].copy()
    scam_data[0] = seoul_gus # 지역명 강제 매핑
    
    # 컬럼명 설정 (1, 2, 3번 컬럼은 2023, 2024, 2025년 데이터)
    scam_data.columns = ['gu_std', '2023', '2024', '2025']
    
    # 숫자 데이터 정제
    for col in ['2023', '2024', '2025']:
        scam_data[col] = scam_data[col].astype(str).str.replace('"', '').str.replace(',', '').str.strip()
        scam_data[col] = pd.to_numeric(scam_data[col], errors='coerce').fillna(0).astype(int)
    
    # Wide to Long Format 변환
    scam_long = scam_data.melt(id_vars='gu_std', var_name='year', value_name='사기발생빈도')
    scam_long['year'] = scam_long['year'].astype(int)
    
    # 2. 전세가율 데이터 로드
    ratio_path = os.path.join(DATASETS_CLEANED, "jeonse_ratio_by_gu_year_area.csv")
    ratio_df = pd.read_csv(ratio_path)
    ratio_agg = ratio_df.groupby(['gu_std', 'year'])['jeonse_ratio_pct'].mean().reset_index()
    
    print("Scam Data Sample:")
    print(scam_long.head())
    print("Ratio Data Sample:")
    print(ratio_agg.head())
    
    return scam_long, ratio_agg

def analyze_and_visualize(scam_long, ratio_agg):
    # 데이터 병합
    merged_df = pd.merge(scam_long, ratio_agg, on=['gu_std', 'year'], how='inner')
    
    if merged_df.empty:
        print("병합된 데이터가 없습니다. 구(Gu) 명칭이나 연도(Year)를 확인하세요.")
        return None
    
    # 상관관계 계산
    corr, p_value = pearsonr(merged_df['jeonse_ratio_pct'], merged_df['사기발생빈도'])
    print(f"상관계수: {corr:.4f}, P-value: {p_value:.4f}")
    
    # 시각화 1: 산점도 및 회귀선
    plt.figure(figsize=(10, 6))
    sns.regplot(data=merged_df, x='jeonse_ratio_pct', y='사기발생빈도', 
                scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    plt.title('전세가율과 전세 사기 발생 빈도 간의 상관관계')
    plt.xlabel('평균 전세가율 (%)')
    plt.ylabel('전세 사기 발생 건수')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    image_path = os.path.join(IMAGES_DIR, "jeonse_ratio_scam_correlation.png")
    plt.savefig(image_path)
    plt.close()
    
    # 시각화 2: 연도별 산점도
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=merged_df, x='jeonse_ratio_pct', y='사기발생빈도', hue='year', palette='viridis', s=100)
    plt.title('연도별 전세가율 vs 전세 사기 빈도')
    plt.xlabel('평균 전세가율 (%)')
    plt.ylabel('전세 사기 발생 건수')
    plt.legend(title='연도')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    year_image_path = os.path.join(IMAGES_DIR, "jeonse_ratio_scam_by_year.png")
    plt.savefig(year_image_path)
    plt.close()
    
    return merged_df, corr, p_value

def generate_report(merged_df, corr, p_value):
    report_path = os.path.join(REPORT_DIR, "rental_scam_correlation_report.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 전세 사기 빈도와 전세가율 상관관계 분석 보고서\n\n")
        f.write("## 1. 분석 개요\n")
        f.write("- **목적**: 서울시 구별 전세가율 수준과 전세 사기 발생 빈도 사이의 상관관계를 파악하여, 전세가율이 높은 지역이 전세 사기에 취약한지 분석합니다.\n")
        f.write("- **분석 대상**: 2023년 ~ 2025년 서울시 구별 데이터\n\n")
        
        f.write("## 2. 분석 결과\n")
        f.write(f"- **피어슨 상관계수**: {corr:.4f}\n")
        f.write(f"- **P-value**: {p_value:.4e}\n\n")
        
        interpretation = ""
        if p_value < 0.05:
            if corr > 0:
                interpretation = "통계적으로 유의미한 **양(+)의 상관관계**가 관찰되었습니다. 즉, 전세가율이 높을수록 전세 사기 발생 발생 건수가 증가하는 경향이 있습니다."
            else:
                interpretation = "통계적으로 유의미한 **음(-)의 상관관계**가 관찰되었습니다."
        else:
            interpretation = "통계적으로 유의미한 상관관계가 관찰되지 않았습니다 (P-value > 0.05)."
        
        f.write(f"### 해석\n{interpretation}\n\n")
        
        f.write("## 3. 시각화\n")
        f.write("### 전세가율과 전세 사기 빈도 산점도 (전체)\n")
        f.write("![상관관계 산점도](../images/jeonse_ratio_scam_correlation.png)\n\n")
        f.write("### 연도별 추이\n")
        f.write("![연도별 산점도](../images/jeonse_ratio_scam_by_year.png)\n\n")
        
        f.write("## 4. 데이터 요약 (상위 5건)\n")
        summary_df = merged_df.sort_values(by='사기발생빈도', ascending=False).head(10)
        f.write(summary_df.to_markdown(index=False))
        f.write("\n")

if __name__ == "__main__":
    scam_long, ratio_agg = load_data()
    result = analyze_and_visualize(scam_long, ratio_agg)
    if result:
        merged_df, corr, p_value = result
        generate_report(merged_df, corr, p_value)
        print("분석 완료 및 보고서 생성됨.")
