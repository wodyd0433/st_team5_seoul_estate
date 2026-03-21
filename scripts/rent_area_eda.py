import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib
import os
import glob

# 디렉토리 설정
BASE_DIR = r"c:\Users\wodyd\OneDrive\PythonWorkspace\icb7\korea_estate"
DATASETS_RAW = os.path.join(BASE_DIR, "datasets", "raw")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
REPORT_DIR = os.path.join(BASE_DIR, "report")

# 필요한 디렉토리 생성
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

def load_and_combine_data():
    # apartment_rent_transactions_*.csv 파일 목록 가져오기
    file_pattern = os.path.join(DATASETS_RAW, "apartment_rent_transactions_*.csv")
    file_list = glob.glob(file_pattern)
    print(f"찾은 데이터 파일: {len(file_list)}개")
    
    dfs = []
    for file in file_list:
        try:
            # 데이터 로드
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"로드 완료: {os.path.basename(file)} ({len(df)}행)")
        except Exception as e:
            print(f"파일 로드 실패: {file}, 오류: {e}")
            
    if not dfs:
        return None
    
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def preprocess_data(df):
    # 전용면적_m2를 평수로 변환
    df['평수'] = df['전용면적_m2'] / 3.3
    
    # 평수 라운딩 (소수점 1자리)
    df['평수_반올림'] = df['평수'].round(1)
    
    # 평수 구간 설정
    def get_area_category(py):
        if py < 10: return '10평 미만'
        elif py < 20: return '10-20평'
        elif py < 30: return '20-30평'
        elif py < 40: return '30-40평'
        elif py < 50: return '40-50평'
        else: return '50평 이상'
        
    df['평수구간'] = df['평수'].apply(get_area_category)
    
    # 이상치 제거 (0 이하이거나 너무 큰 평수 - 200평 초과 등)
    initial_count = len(df)
    df = df[(df['평수'] > 0) & (df['평수'] <= 200)].copy()
    print(f"이상치 제거 후 데이터 수: {len(df)} / {initial_count}")
    
    return df

def analyze_and_visualize(df):
    # 기초 통계량
    stats = df['평수'].describe()
    print("\n평수 기초 통계량:")
    print(stats)
    
    # 1. 히스토그램 (전체 분포)
    plt.figure(figsize=(12, 6))
    sns.histplot(df['평수'], bins=50, kde=True, color='skyblue')
    plt.axvline(df['평수'].mean(), color='red', linestyle='--', label=f'평균: {df["평수"].mean():.1f}평')
    plt.axvline(df['평수'].median(), color='green', linestyle='-', label=f'중앙값: {df["평수"].median():.1f}평')
    plt.title('서울시 아파트 전세 거래 평수 분포 (히스토그램)')
    plt.xlabel('평수')
    plt.ylabel('거래 건수')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(IMAGES_DIR, "rent_area_distribution_hist.png"))
    plt.close()
    
    # 2. 평수구간별 파이 차트
    plt.figure(figsize=(10, 8))
    area_counts = df['평수구간'].value_counts().reindex(['10평 미만', '10-20평', '20-30평', '30-40평', '40-50평', '50평 이상'])
    area_counts.plot.pie(autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
    plt.title('평수 구간별 거래 비중')
    plt.ylabel('')
    plt.savefig(os.path.join(IMAGES_DIR, "rent_area_bins_pie.png"))
    plt.close()
    
    # 3. 자치구별 상위 10개 평수 분포 (Boxplot)
    top_10_gus = df['구'].value_counts().head(10).index
    df_top_gus = df[df['구'].isin(top_10_gus)]
    
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df_top_gus, x='구', y='평수', order=top_10_gus, palette='Set3')
    plt.title('주요 자치구별 거래 평수 분포')
    plt.xticks(rotation=45)
    plt.xlabel('자치구')
    plt.ylabel('평수')
    plt.savefig(os.path.join(IMAGES_DIR, "rent_area_by_gu_boxplot.png"))
    plt.close()
    
    # 가장 빈번한 평수 Top 10 (m2 기준 원래 값들이 특정 규격인 경우가 많음)
    common_areas = df['전용면적_m2'].value_counts().head(10)
    common_areas_df = common_areas.reset_index()
    common_areas_df.columns = ['전용면적_m2', '거래건수']
    common_areas_df['평수'] = (common_areas_df['전용면적_m2'] / 3.3).round(2)
    
    return stats, common_areas_df

def generate_report(stats, common_areas_df, df):
    report_path = os.path.join(REPORT_DIR, "apartment_rent_area_eda_report.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 서울시 아파트 전세 거래 평수 분포 EDA 보고서\n\n")
        f.write("## 1. 데이터 개요\n")
        f.write(f"- **분석 대상**: 2021년 ~ 2026년 서울시 아파트 전세 거래 내역\n")
        f.write(f"- **총 거래 건수**: {len(df):,}건\n\n")
        
        f.write("## 2. 평수 기초 통계량\n")
        stats_md = stats.to_frame().T.to_markdown(index=False)
        f.write(f"{stats_md}\n\n")
        f.write(f"- **평균 평수**: {stats['mean']:.2f}평\n")
        f.write(f"- **중앙값 (50%)**: {stats['50%']:.2f}평\n")
        f.write(f"- **최대 평수**: {stats['max']:.2f}평 (이상치 제외 후)\n\n")
        
        f.write("## 3. 시각화 분석\n")
        f.write("### 전체 평수 분포 히스토그램\n")
        f.write("![평수 히스토그램](../images/rent_area_distribution_hist.png)\n\n")
        f.write("> **해석**: 대부분의 거래가 20~30평 대역에 집중되어 있는 것을 확인할 수 있습니다.\n\n")
        
        f.write("### 평수 구간별 비중\n")
        f.write("![평수 비중 파이차트](../images/rent_area_bins_pie.png)\n\n")
        
        f.write("### 주요 자치구별 평수 분포 현황\n")
        f.write("![자치구별 박스플롯](../images/rent_area_by_gu_boxplot.png)\n\n")
        
        f.write("## 4. 가장 빈번하게 거래되는 전용면적 Top 10\n")
        f.write(common_areas_df.to_markdown(index=False))
        f.write("\n\n")
        f.write("> **참고**: 84.9m² 내외(약 25.7평)와 59.9m² 내외(약 18.1평)가 가장 압도적인 거래량을 보입니다.\n")

if __name__ == "__main__":
    raw_df = load_and_combine_data()
    if raw_df is not None:
        clean_df = preprocess_data(raw_df)
        stats, top_10 = analyze_and_visualize(clean_df)
        generate_report(stats, top_10, clean_df)
        print("\nEDA 완료 및 보고서 생성됨.")
    else:
        print("데이터를 찾을 수 없습니다.")
