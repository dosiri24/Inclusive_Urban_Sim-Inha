"""Calculate demographic ratios from Michuhol-gu population data."""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "datas"


def calc_age_group_ratios():
    """Calculate population ratio by age groups (20+ only, for debate participation)."""
    df = pd.read_csv(DATA_DIR / "인천광역시 미추홀구_1세간격 인구통계_20250831.csv", encoding="cp949")

    age_groups = {
        "20대": range(20, 30),
        "30대": range(30, 40),
        "40대": range(40, 50),
        "50대": range(50, 60),
        "60대": range(60, 70),
        "70대 이상": range(70, 120),
    }

    ages = df["연령"].str.extract(r"(\d+)", expand=False).astype(float)
    total_20_plus = df[ages >= 20]["인구수"].sum()

    results = []
    for group_name, age_range in age_groups.items():
        group_pop = df[ages.isin(age_range)]["인구수"].sum()
        ratio = group_pop / total_20_plus
        results.append(f"{group_name}: {ratio:.4f}")

    return results


def calc_disability_ratio():
    """Calculate disability ratio in the region."""
    df_disabled = pd.read_csv(DATA_DIR / "연령 및 성별 장애인 인구 - 시군구.csv", encoding="cp949", header=1)
    df_total = pd.read_csv(DATA_DIR / "연령 및 성별 인구 – 읍면동.csv", encoding="cp949", header=1)

    disabled_total = int(df_disabled[df_disabled["연령별"] == "합계"].iloc[0, 2])
    total_pop = int(df_total[df_total["연령별"] == "합계"]["총인구(명)"].values[0])

    ratio = disabled_total / total_pop
    return f"장애인 비율: {ratio:.4f}"


def calc_gender_ratio():
    """Calculate male/female ratio."""
    df = pd.read_csv(DATA_DIR / "연령 및 성별 인구 – 읍면동.csv", encoding="cp949", header=1)

    total_row = df[df["연령별"] == "합계"].iloc[0]
    male = int(total_row["총인구_남자(명)"])
    female = int(total_row["총인구_여자(명)"])
    total = int(total_row["총인구(명)"])

    return [
        f"남성 비율: {male / total:.4f}",
        f"여성 비율: {female / total:.4f}",
    ]


def calc_foreigner_ratio():
    """Calculate foreigner ratio."""
    df = pd.read_csv(DATA_DIR / "연령 및 성별 인구 – 읍면동.csv", encoding="cp949", header=1)

    total_row = df[df["연령별"] == "합계"].iloc[0]
    total = int(total_row["총인구(명)"])
    domestic = int(total_row["내국인(명)"])
    foreigner = total - domestic

    return f"외국인 비율: {foreigner / total:.4f}"


def calc_household_ratios():
    """Calculate household type ratios."""
    df = pd.read_csv(DATA_DIR / "성, 연령 및 세대구성별 인구 - 시군구.csv", encoding="cp949", header=1)

    total_row = df[df["연령별"] == "합계"].iloc[0]
    total_household_pop = int(total_row["일반가구원"])

    one_gen = int(total_row["1세대가구-계"])
    two_gen = int(total_row["2세대가구-계"])
    three_gen = int(total_row["3세대가구-계"])
    four_plus_gen = int(total_row["4세대이상 가구"])
    single = int(total_row["1인가구"])
    non_relative = int(total_row["비친족가구"])

    return [
        f"1세대가구 비율: {one_gen / total_household_pop:.4f}",
        f"2세대가구 비율: {two_gen / total_household_pop:.4f}",
        f"3세대가구 비율: {three_gen / total_household_pop:.4f}",
        f"4세대이상가구 비율: {four_plus_gen / total_household_pop:.4f}",
        f"1인가구 비율: {single / total_household_pop:.4f}",
        f"비친족가구 비율: {non_relative / total_household_pop:.4f}",
    ]


def calc_elderly_living_alone_ratio():
    """Calculate elderly (65+) living alone ratio among all single-person households."""
    df = pd.read_csv(DATA_DIR / "성, 연령 및 세대구성별 인구 - 시군구.csv", encoding="cp949", header=1)

    total_single = int(df[df["연령별"] == "합계"]["1인가구"].values[0])

    elderly_ages = ["65~69세", "70~74세", "75~79세", "80~84세", "85세이상"]
    elderly_single = 0
    for age in elderly_ages:
        val = df[df["연령별"] == age]["1인가구"].values[0]
        if val != "X":
            elderly_single += int(val)

    return f"독거노인 비율 (1인가구 중): {elderly_single / total_single:.4f}"


def calc_occupation_ratios():
    """Calculate occupation ratios combining employed and non-economic population (based on 360k)."""
    df_emp = pd.read_csv(DATA_DIR / " 시군구:산업별 취업자(10차).csv", encoding="cp949", header=1)
    df_emp["산업별"] = df_emp["산업별"].str.strip()

    df_inactive = pd.read_csv(
        DATA_DIR / "시군구:활동상태별 비경제활동인구 (미추홀구)(2024).csv",
        encoding="cp949",
    )

    total = 360  # 219 (employed) + 141 (inactive) in thousands

    results = []

    industries = [
        ("농업/임업/어업", "농업, 임업 및 어업 (A)"),
        ("광/제조업", "광·제조업(B,C)"),
        ("건설업", "건설업 (F)"),
        ("도소매/숙박음식업", "도소매·숙박음식업(G,I)"),
        ("전기/운수/통신/금융", "전기·운수·통신·금융(D,H,J,K)"),
        ("사업/개인/공공서비스", "사업·개인·공공서비스 및 기타(E,L~U)"),
    ]
    for name, col_name in industries:
        val = float(df_emp[df_emp["산업별"] == col_name]["취업자 (천명)"].values[0])
        results.append(f"{name}: {val / total:.4f}")

    inactive_categories = [
        ("주부", "육아, 가사"),
        ("학생", "재학,진학준비(정규교육기관 통학, 입시학원 통학, 진학 준비 포함)"),
        ("은퇴", "연로"),
        ("기타(무직 등)", "기타(취업준비, 심신장애, 군입대대기, 결혼준비, 쉬었음 등)"),
    ]
    for name, col_name in inactive_categories:
        val = float(df_inactive[df_inactive["활동상태별"] == col_name]["2024.2.2"].values[0])
        results.append(f"{name}: {val / total:.4f}")

    return results


def calc_housing_ownership_ratio():
    """Calculate housing ownership vs non-ownership ratio."""
    df = pd.read_csv(DATA_DIR / "거주지역별 주택소유 및 무주택 가구수.csv", encoding="cp949", header=1)

    row = df.iloc[0]
    total = int(row["총 가구(일반가구)"])
    owned = int(row["주택소유 가구"])
    not_owned = int(row["무주택 가구"])

    return [
        f"자가 비율: {owned / total:.4f}",
        f"비자가 비율: {not_owned / total:.4f}",
    ]


def calc_income_ratios():
    """Calculate income bracket ratios."""
    df = pd.read_csv(DATA_DIR / "전국 시군구 단위 소득 구간대별 주민 비율 코리아크레딧뷰로.csv", encoding="utf-8")

    row = df.iloc[0]
    results = [
        f"2천만원대: {row['소득2천만원주민비율']:.2f}",
        f"3천만원대: {row['소득3천만원주민비율']:.2f}",
        f"4천만원대: {row['소득4천만원주민비율']:.2f}",
        f"5천만원대: {row['소득5천만원주민비율']:.2f}",
        f"6천만원대: {row['소득6천만원주민비율']:.2f}",
        f"7천만원대: {row['소득7천만원주민비율']:.2f}",
        f"7천만원 이상: {row['소득7천만원이상주민비율']:.2f}",
    ]
    return results


def calc_housing_type_ratios():
    """Calculate housing type ratios for Juan2-dong area."""
    df = pd.read_csv(
        DATA_DIR / "주택의 종류별 주택 - 읍면동(연도 끝자리 0, 5), 시군구(그 외 연도)(주안2동)(2020).csv",
        encoding="cp949",
        header=1,
    )

    row = df.iloc[0]
    total = int(row["주택"])

    apartment = int(row["아파트"])
    multi_family = int(row["다세대주택"]) + int(row["연립주택"])
    detached = int(row["단독주택-계"])
    other = int(row["비주거용 건물 내 주택"])

    return [
        f"아파트: {apartment / total:.4f}",
        f"다세대/연립: {multi_family / total:.4f}",
        f"단독: {detached / total:.4f}",
        f"기타: {other / total:.4f}",
    ]


def calc_housing_tenure_ratios():
    """Calculate housing tenure ratios (owned/jeonse/monthly rent)."""
    df = pd.read_csv(
        DATA_DIR / "가구주의 연령별:점유형태별 가구(일반가구)-시군구(2020).csv",
        encoding="cp949",
        header=1,
    )

    row = df[df["연령별"] == "합계"].iloc[0]
    total = int(row["일반가구"])

    owned = int(row["자가"])
    jeonse = int(row["전세(월세없음)"])
    monthly_deposit = int(row["보증금 있는 월세"])
    monthly_no_deposit = int(row["보증금 없는 월세"])
    saglse = int(row["사글세"])
    free = int(row["무상(관사 사택 등)"])

    monthly_total = monthly_deposit + monthly_no_deposit + saglse

    return [
        f"자가: {owned / total:.4f}",
        f"전세: {jeonse / total:.4f}",
        f"월세: {monthly_total / total:.4f}",
        f"무상: {free / total:.4f}",
    ]


def calc_residence_duration_ratios():
    """Calculate residence duration ratios grouped into 4 brackets (Incheon-wide)."""
    df = pd.read_excel(
        DATA_DIR / "(일반가구)행정구역별 현재주택 거주기간 (인천)(2024).xlsx",
        header=1,
    )

    row = df[df["구분(1)"] == "인천"].iloc[0]

    r_1_5 = (row["1년 미만"] + row["1년~2년"] + row["2년~3년"] + row["3년~5년"]) / 100
    r_5_10 = row["5년~10년"] / 100
    r_10_20 = (row["10년~15년"] + row["15년~20년"]) / 100
    r_20_plus = (row["20년~25년"] + row["25년 이상"]) / 100

    return [
        f"1-5년: {r_1_5:.4f} (평균 3년)",
        f"5-10년: {r_5_10:.4f} (평균 7년)",
        f"10-20년: {r_10_20:.4f} (평균 15년)",
        f"20년 이상: {r_20_plus:.4f} (평균 25년)",
    ]


if __name__ == "__main__":
    print("[연령대별 구성 비율]")
    for line in calc_age_group_ratios():
        print(line)

    print("\n[장애인 비율]")
    print(calc_disability_ratio())

    print("\n[성별 비율]")
    for line in calc_gender_ratio():
        print(line)

    print("\n[외국인 비율]")
    print(calc_foreigner_ratio())

    print("\n[가구유형 비율]")
    for line in calc_household_ratios():
        print(line)

    print("\n[독거노인 비율]")
    print(calc_elderly_living_alone_ratio())

    print("\n[직업 비율 - 360천명 기준]")
    for line in calc_occupation_ratios():
        print(line)

    print("\n[주택소유 비율]")
    for line in calc_housing_ownership_ratio():
        print(line)

    print("\n[소득 구간대별 비율]")
    for line in calc_income_ratios():
        print(line)

    print("\n[주택 종류별 비율 - 주안2동]")
    for line in calc_housing_type_ratios():
        print(line)

    print("\n[점유형태별 비율]")
    for line in calc_housing_tenure_ratios():
        print(line)

    print("\n[거주기간별 비율 - 인천]")
    for line in calc_residence_duration_ratios():
        print(line)
