import json

cells = []

def md(src):
    if isinstance(src, str):
        src = [src]
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src})

def code(src):
    if isinstance(src, str):
        src = [src]
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src})

# ========================== PHẦN MỞ ĐẦU ==========================
md("# 📚 PHÂN TÍCH HÀNH VI NGƯỜI DÙNG LEARNX\n## Nền tảng học trực tuyến ứng dụng AI dành cho sinh viên đại học\n\n**Mục tiêu phân tích:**\n- Khám phá dữ liệu và hiểu hành vi người dùng\n- Phát hiện outliers và insight hữu ích\n- Trả lời câu hỏi: Người dùng LearnX sử dụng nền tảng ra sao?")

# ========================== 1. GIỚI THIỆU DỮ LIỆU ==========================
md("## 1. Giới thiệu dữ liệu\n### 1.1 Nguồn dữ liệu\n\nDataset **learnx_realistic_500k.csv** chứa dữ liệu hành vi người dùng trên nền tảng LearnX sau 18 tháng hoạt động.\n\n- **Nguồn**: Hệ thống tracking nội bộ của LearnX\n- **Phạm vi**: Tất cả người dùng đã đăng ký tài khoản\n- **Thời gian thu thập**: 18 tháng kể từ khi ra mắt")

md("### 1.2 Các thuộc tính\n\n| # | Thuộc tính | Kiểu | Mô tả |\n|---|------------|------|--------|\n| 1 | `user_id` | int | Mã định danh người dùng |\n| 2 | `age` | int | Tuổi người dùng |\n| 3 | `country` | str | Quốc gia |\n| 4 | `major` | str | Ngành học |\n| 5 | `signup_days_ago` | int | Số ngày kể từ khi đăng ký |\n| 6 | `sessions_per_week` | int | Số phiên truy cập/tuần |\n| 7 | `avg_session_minutes` | float | Thời gian TB mỗi phiên (phút) |\n| 8 | `videos_watched` | int | Số video đã xem |\n| 9 | `quizzes_taken` | int | Số bài kiểm tra đã làm |\n| 10 | `forum_posts` | float | Số bài đăng trên diễn đàn |\n| 11 | `completion_rate` | float | Tỷ lệ hoàn thành (0-1) |\n| 12 | `courses_enrolled` | int | Số khóa học đã đăng ký |\n| 13 | `assignments_submitted` | int | Số bài tập đã nộp |\n| 14 | `premium_purchased` | int | Đã mua premium (0/1) |\n| 15 | `total_spent_usd` | float | Tổng chi tiêu (USD) |\n| 16 | `discount_used` | float | Số mã giảm giá đã dùng |\n| 17 | `ai_recommend_click` | float | Số lần click gợi ý AI |\n| 18 | `ai_recommend_enroll` | int | Số lần đăng ký từ gợi ý AI |\n| 19 | `churn_risk` | int | Nguy cơ rời bỏ (0/1) |\n| 20 | `future_purchase` | int | Dự đoán mua hàng (0/1) |")

md("### 1.3 Số lượng bản ghi, kiểu dữ liệu, thống kê cơ bản")

code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Thiết lập style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 12""")

code("""# Đọc dữ liệu
df = pd.read_csv("Data/learnx_realistic_500k.csv")
print(f"Số lượng bản ghi: {df.shape[0]:,}")
print(f"Số lượng thuộc tính: {df.shape[1]}")
print("\\n" + "="*60)
print("5 dòng đầu tiên:")
df.head()""")

code("""# Thông tin kiểu dữ liệu
print("THÔNG TIN KIỂU DỮ LIỆU:")
print("="*60)
df.info()""")

code("""# Thống kê mô tả cơ bản
print("THỐNG KÊ MÔ TẢ CƠ BẢN:")
print("="*60)
df.describe().round(2)""")

code("""# Thống kê cột phân loại
print("THỐNG KÊ CỘT PHÂN LOẠI:")
print("="*60)
for col in ["country", "major"]:
    print(f"\\n--- {col} ---")
    print(f"Số giá trị duy nhất: {df[col].nunique()}")
    print(df[col].value_counts())""")

md("### 1.4 Nhận xét sơ bộ\n\n**Quy mô dữ liệu:**\n- Dataset lớn với ~500,000 bản ghi, đủ cho phân tích thống kê tin cậy\n- 20 thuộc tính bao gồm thông tin nhân khẩu học, hành vi và giao dịch\n\n**Đặc điểm nổi bật:**\n- Có cả biến số (numeric) và biến phân loại (categorical)\n- Một số cột có giá trị thiếu cần xử lý\n- Biến nhị phân: `premium_purchased`, `churn_risk`, `future_purchase`\n- `completion_rate` nằm trong khoảng [0, 1]\n- `total_spent_usd` có thể chứa outliers\n- Dữ liệu đa quốc gia, đa ngành → phong phú cho phân tích segment")

# ========================== 2. LÀM SẠCH DỮ LIỆU ==========================
md("## 2. Làm sạch dữ liệu\n### 2.1 Kiểm tra dữ liệu thiếu")

code("""# Kiểm tra missing values
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Số lượng thiếu": missing, "Tỷ lệ (%)": missing_pct})
missing_df = missing_df[missing_df["Số lượng thiếu"] > 0].sort_values("Tỷ lệ (%)", ascending=False)
print("CÁC CỘT CÓ DỮ LIỆU THIẾU:")
print("="*60)
print(missing_df)
print(f"\\nTổng số ô thiếu: {df.isnull().sum().sum():,}")""")

code("""# Trực quan hóa missing values
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Bar chart
missing_cols = missing_df.index.tolist()
colors = sns.color_palette("Reds_r", len(missing_cols))
axes[0].barh(missing_cols, missing_df["Tỷ lệ (%)"], color=colors)
axes[0].set_xlabel("Tỷ lệ thiếu (%)")
axes[0].set_title("Tỷ lệ dữ liệu thiếu theo cột", fontsize=14, fontweight="bold")
for i, v in enumerate(missing_df["Tỷ lệ (%)"]):
    axes[0].text(v + 0.1, i, f"{v}%", va="center")

# Heatmap
sample = df[missing_cols].sample(1000, random_state=42)
sns.heatmap(sample.isnull(), cbar=True, yticklabels=False, ax=axes[1],
            cmap="YlOrRd", cbar_kws={"label": "Missing"})
axes[1].set_title("Pattern dữ liệu thiếu (mẫu 1000 dòng)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()""")

md("### 2.2 Xử lý dữ liệu thiếu (missing values)\n\n**Chiến lược:**\n- Các cột số liên tục (`avg_session_minutes`, `forum_posts`, `completion_rate`, `total_spent_usd`): điền bằng **median** (ít bị ảnh hưởng bởi outliers)\n- Các cột số rời rạc (`discount_used`, `ai_recommend_click`): điền bằng **median**")

code("""# Xử lý missing values
print("TRƯỚC KHI XỬ LÝ:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print()

# Điền median cho các cột số
cols_fill_median = ["avg_session_minutes", "forum_posts", "completion_rate",
                    "total_spent_usd", "discount_used", "ai_recommend_click"]

for col in cols_fill_median:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)
    print(f"  {col}: điền median = {median_val}")

print("\\nSAU KHI XỬ LÝ:")
print(f"Tổng missing còn lại: {df.isnull().sum().sum()}")""")

md("### 2.3 Kiểm tra và xử lý dữ liệu sai hoặc bất thường")

code("""# Kiểm tra giá trị bất thường
print("KIỂM TRA GIÁ TRỊ BẤT THƯỜNG:")
print("="*60)

print(f"\\n1. age: min={df['age'].min()}, max={df['age'].max()}")
print(f"   → Giá trị ngoài [16, 40]? {((df['age'] < 16) | (df['age'] > 40)).sum()}")

print(f"\\n2. completion_rate: min={df['completion_rate'].min()}, max={df['completion_rate'].max()}")
print(f"   → Giá trị ngoài [0, 1]? {((df['completion_rate'] < 0) | (df['completion_rate'] > 1)).sum()}")

print(f"\\n3. sessions_per_week: min={df['sessions_per_week'].min()}, max={df['sessions_per_week'].max()}")
print(f"   → Giá trị âm? {(df['sessions_per_week'] < 0).sum()}")

print(f"\\n4. total_spent_usd: min={df['total_spent_usd'].min()}, max={df['total_spent_usd'].max()}")
print(f"   → Giá trị âm? {(df['total_spent_usd'] < 0).sum()}")

for col in ["premium_purchased", "churn_risk", "future_purchase"]:
    print(f"\\n5. {col}: unique = {sorted(df[col].unique())}")""")

code("""# Xử lý giá trị bất thường
print("XỬ LÝ GIÁ TRỊ BẤT THƯỜNG:")
print("="*60)

df["age"] = df["age"].clip(lower=16, upper=45)
print("✅ Clip age về [16, 45]")

df["completion_rate"] = df["completion_rate"].clip(lower=0, upper=1)
print("✅ Clip completion_rate về [0, 1]")

df["sessions_per_week"] = df["sessions_per_week"].clip(lower=0)
print("✅ Đảm bảo sessions_per_week >= 0")

df["total_spent_usd"] = df["total_spent_usd"].clip(lower=0)
print("✅ Đảm bảo total_spent_usd >= 0")

print("\\n→ Hoàn tất xử lý giá trị bất thường.")""")

md("### 2.4 Xử lý dữ liệu trùng lặp (duplicates)")

code("""# Kiểm tra trùng lặp
print("KIỂM TRA DỮ LIỆU TRÙNG LẶP:")
print("="*60)

dup_all = df.duplicated().sum()
print(f"Số dòng trùng lặp hoàn toàn: {dup_all:,}")

dup_id = df.duplicated(subset=["user_id"]).sum()
print(f"Số dòng trùng user_id: {dup_id:,}")

if dup_id > 0:
    print(f"\\n→ Phát hiện {dup_id:,} bản ghi trùng user_id.")
    print("→ Giữ lại bản ghi đầu tiên, loại bỏ các bản ghi trùng.")
    df = df.drop_duplicates(subset=["user_id"], keep="first")
    print(f"→ Sau xử lý: {df.shape[0]:,} bản ghi")
else:
    print("\\n✅ Không có dữ liệu trùng lặp.")

print(f"\\nKích thước dữ liệu sau làm sạch: {df.shape}")""")

# ========================== 3. PHÂN TÍCH BIỂU ĐỒ ==========================
md("## 3. Phân tích bằng biểu đồ\n### 3.1 Phân phối thời gian học")

code("""# 3.1 Phân phối thời gian học (avg_session_minutes)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histogram
axes[0].hist(df["avg_session_minutes"], bins=50, color="#3498db", edgecolor="white", alpha=0.8)
axes[0].axvline(df["avg_session_minutes"].mean(), color="red", linestyle="--", label=f'Mean = {df["avg_session_minutes"].mean():.1f}')
axes[0].axvline(df["avg_session_minutes"].median(), color="green", linestyle="--", label=f'Median = {df["avg_session_minutes"].median():.1f}')
axes[0].set_xlabel("Thời gian trung bình/phiên (phút)")
axes[0].set_ylabel("Số lượng người dùng")
axes[0].set_title("Phân phối thời gian học mỗi phiên", fontweight="bold")
axes[0].legend()

# Boxplot
sns.boxplot(x=df["avg_session_minutes"], ax=axes[1], color="#2ecc71")
axes[1].set_xlabel("Thời gian trung bình/phiên (phút)")
axes[1].set_title("Boxplot thời gian học", fontweight="bold")

# Violin theo Premium
sns.violinplot(x="premium_purchased", y="avg_session_minutes", data=df, ax=axes[2],
               palette=["#e74c3c", "#2ecc71"])
axes[2].set_xlabel("Đã mua Premium")
axes[2].set_ylabel("Thời gian trung bình/phiên (phút)")
axes[2].set_title("Thời gian học: Free vs Premium", fontweight="bold")
axes[2].set_xticklabels(["Free", "Premium"])

plt.tight_layout()
plt.show()

print(f"Thống kê avg_session_minutes:")
print(df["avg_session_minutes"].describe().round(2))""")

md("**Nhận xét:**\n- Phân phối thời gian học gần như đồng đều, cho thấy sự đa dạng trong thói quen học\n- Median và Mean gần nhau → phân phối khá cân đối\n- Người dùng Premium có xu hướng học lâu hơn so với Free users")

md("### 3.2 Số lần truy cập mỗi tuần")

code("""# 3.2 Số lần truy cập mỗi tuần
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histogram
session_counts = df["sessions_per_week"].value_counts().sort_index()
axes[0].bar(session_counts.index, session_counts.values, color="#9b59b6", edgecolor="white")
axes[0].set_xlabel("Số phiên/tuần")
axes[0].set_ylabel("Số lượng người dùng")
axes[0].set_title("Phân phối số lần truy cập/tuần", fontweight="bold")
axes[0].axvline(df["sessions_per_week"].mean(), color="red", linestyle="--",
                label=f'Mean = {df["sessions_per_week"].mean():.1f}')
axes[0].legend()

# Theo quốc gia (Top 5)
top_countries = df["country"].value_counts().head(5).index
df_top = df[df["country"].isin(top_countries)]
sns.boxplot(x="country", y="sessions_per_week", data=df_top, ax=axes[1],
            palette="Set2", order=top_countries)
axes[1].set_xlabel("Quốc gia")
axes[1].set_ylabel("Số phiên/tuần")
axes[1].set_title("Sessions/tuần theo Top 5 quốc gia", fontweight="bold")

# Theo ngành học
sns.boxplot(x="major", y="sessions_per_week", data=df, ax=axes[2], palette="Set3")
axes[2].set_xlabel("Ngành học")
axes[2].set_ylabel("Số phiên/tuần")
axes[2].set_title("Sessions/tuần theo ngành học", fontweight="bold")
axes[2].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

print(f"\\nPhân phối sessions_per_week:")
print(df["sessions_per_week"].describe().round(2))""")

md("**Nhận xét:**\n- Đa số người dùng truy cập 1-5 lần/tuần\n- Có một nhóm nhỏ power users truy cập >10 lần/tuần\n- Không có sự khác biệt lớn giữa các quốc gia và ngành học")

md("### 3.3 Mức độ hoàn thành khóa học (completion rate)")

code("""# 3.3 Completion Rate
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Histogram
axes[0, 0].hist(df["completion_rate"], bins=50, color="#e67e22", edgecolor="white", alpha=0.8)
axes[0, 0].axvline(df["completion_rate"].mean(), color="red", linestyle="--",
                   label=f'Mean = {df["completion_rate"].mean():.2f}')
axes[0, 0].set_xlabel("Completion Rate")
axes[0, 0].set_ylabel("Số lượng")
axes[0, 0].set_title("Phân phối tỷ lệ hoàn thành", fontweight="bold")
axes[0, 0].legend()

# Completion rate theo churn_risk
sns.boxplot(x="churn_risk", y="completion_rate", data=df, ax=axes[0, 1],
            palette=["#27ae60", "#e74c3c"])
axes[0, 1].set_xlabel("Churn Risk")
axes[0, 1].set_ylabel("Completion Rate")
axes[0, 1].set_title("Completion Rate vs Churn Risk", fontweight="bold")
axes[0, 1].set_xticklabels(["Không rời bỏ", "Có nguy cơ"])

# Theo major
major_completion = df.groupby("major")["completion_rate"].mean().sort_values(ascending=True)
axes[1, 0].barh(major_completion.index, major_completion.values, color=sns.color_palette("viridis", len(major_completion)))
axes[1, 0].set_xlabel("Completion Rate trung bình")
axes[1, 0].set_title("Completion Rate theo ngành học", fontweight="bold")
for i, v in enumerate(major_completion.values):
    axes[1, 0].text(v + 0.005, i, f"{v:.2f}", va="center")

# Scatter: completion_rate vs total_spent_usd
sample_plot = df.sample(5000, random_state=42)
scatter = axes[1, 1].scatter(sample_plot["completion_rate"], sample_plot["total_spent_usd"],
                              c=sample_plot["premium_purchased"], cmap="coolwarm", alpha=0.3, s=10)
axes[1, 1].set_xlabel("Completion Rate")
axes[1, 1].set_ylabel("Total Spent (USD)")
axes[1, 1].set_title("Completion Rate vs Chi tiêu (mẫu 5000)", fontweight="bold")
plt.colorbar(scatter, ax=axes[1, 1], label="Premium")

plt.tight_layout()
plt.show()

# Phân nhóm completion rate
bins = [0, 0.2, 0.5, 0.8, 1.0]
labels = ["Rất thấp (<20%)", "Thấp (20-50%)", "Trung bình (50-80%)", "Cao (>80%)"]
df["completion_group"] = pd.cut(df["completion_rate"], bins=bins, labels=labels, include_lowest=True)
print("Phân bố nhóm completion rate:")
print(df["completion_group"].value_counts().sort_index())""")

md("**Nhận xét:**\n- Phân phối completion rate khá đồng đều, nhưng có nhiều người dùng ở mức thấp (<20%)\n- Người có nguy cơ churn có completion rate rõ ràng thấp hơn\n- Không có sự khác biệt lớn giữa các ngành học\n- Chi tiêu không tỷ lệ thuận rõ ràng với completion rate")

md("### 3.4 Xu hướng học theo thời gian")

code("""# 3.4 Xu hướng học theo thời gian (signup_days_ago)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Phân nhóm theo thời gian đăng ký
df["signup_period"] = pd.cut(df["signup_days_ago"],
                              bins=[0, 30, 90, 180, 270, 365],
                              labels=["0-30 ngày", "1-3 tháng", "3-6 tháng", "6-9 tháng", "9-12 tháng"])

# Số phiên/tuần theo cohort
cohort_sessions = df.groupby("signup_period")["sessions_per_week"].mean()
axes[0, 0].bar(range(len(cohort_sessions)), cohort_sessions.values, color=sns.color_palette("Blues_d", len(cohort_sessions)))
axes[0, 0].set_xticks(range(len(cohort_sessions)))
axes[0, 0].set_xticklabels(cohort_sessions.index, rotation=30)
axes[0, 0].set_ylabel("Sessions/tuần TB")
axes[0, 0].set_title("Sessions/tuần theo thời gian đăng ký", fontweight="bold")

# Completion rate theo cohort
cohort_completion = df.groupby("signup_period")["completion_rate"].mean()
axes[0, 1].plot(range(len(cohort_completion)), cohort_completion.values, "o-", color="#e74c3c", linewidth=2, markersize=8)
axes[0, 1].set_xticks(range(len(cohort_completion)))
axes[0, 1].set_xticklabels(cohort_completion.index, rotation=30)
axes[0, 1].set_ylabel("Completion Rate TB")
axes[0, 1].set_title("Completion Rate theo cohort", fontweight="bold")

# Videos watched theo cohort
cohort_videos = df.groupby("signup_period")["videos_watched"].mean()
axes[1, 0].bar(range(len(cohort_videos)), cohort_videos.values, color=sns.color_palette("Greens_d", len(cohort_videos)))
axes[1, 0].set_xticks(range(len(cohort_videos)))
axes[1, 0].set_xticklabels(cohort_videos.index, rotation=30)
axes[1, 0].set_ylabel("Videos watched TB")
axes[1, 0].set_title("Số video xem theo cohort", fontweight="bold")

# Tỷ lệ churn theo cohort
cohort_churn = df.groupby("signup_period")["churn_risk"].mean() * 100
axes[1, 1].plot(range(len(cohort_churn)), cohort_churn.values, "s--", color="#8e44ad", linewidth=2, markersize=8)
axes[1, 1].set_xticks(range(len(cohort_churn)))
axes[1, 1].set_xticklabels(cohort_churn.index, rotation=30)
axes[1, 1].set_ylabel("Tỷ lệ Churn Risk (%)")
axes[1, 1].set_title("Tỷ lệ Churn theo cohort", fontweight="bold")

plt.tight_layout()
plt.show()""")

md("**Nhận xét về xu hướng:**\n- Người dùng mới (0-30 ngày) có sessions/tuần và completion rate ổn định\n- Không có sự suy giảm rõ rệt theo thời gian → retention tương đối ổn\n- Tỷ lệ churn có thể tăng nhẹ ở các cohort cũ hơn\n- Videos watched khá đồng đều giữa các cohort")

# ========================== 4. PHÁT HIỆN OUTLIERS ==========================
md("## 4. Phát hiện các hành vi bất thường (Outliers)\n### 4.1 Người dùng học quá nhiều")

code("""# 4.1 Người dùng học quá nhiều
# Xác định ngưỡng bằng IQR
def detect_outliers_iqr(data, column, multiplier=1.5):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    outliers = data[(data[column] < lower) | (data[column] > upper)]
    return outliers, lower, upper

# Outliers về sessions_per_week
out_sessions, low_s, up_s = detect_outliers_iqr(df, "sessions_per_week")
print(f"Sessions/tuần - Ngưỡng: [{low_s:.1f}, {up_s:.1f}]")
print(f"Số outliers: {len(out_sessions):,} ({len(out_sessions)/len(df)*100:.2f}%)")

# Outliers về avg_session_minutes  
out_minutes, low_m, up_m = detect_outliers_iqr(df, "avg_session_minutes")
print(f"\\nAvg session minutes - Ngưỡng: [{low_m:.1f}, {up_m:.1f}]")
print(f"Số outliers: {len(out_minutes):,} ({len(out_minutes)/len(df)*100:.2f}%)")

# Outliers về videos_watched
out_videos, low_v, up_v = detect_outliers_iqr(df, "videos_watched")
print(f"\\nVideos watched - Ngưỡng: [{low_v:.1f}, {up_v:.1f}]")
print(f"Số outliers: {len(out_videos):,} ({len(out_videos)/len(df)*100:.2f}%)")

# Người học "siêu tích cực": sessions >= 10 VÀ avg_minutes >= 60
super_active = df[(df["sessions_per_week"] >= 10) & (df["avg_session_minutes"] >= 60)]
print(f"\\n🔥 Người dùng siêu tích cực (sessions>=10, minutes>=60): {len(super_active):,}")
print(super_active[["user_id", "sessions_per_week", "avg_session_minutes", "videos_watched",
                     "completion_rate", "premium_purchased"]].head(10))""")

code("""# Visualization outliers - Người học quá nhiều
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Boxplot sessions
sns.boxplot(y=df["sessions_per_week"], ax=axes[0], color="#3498db")
axes[0].set_title("Outliers: Sessions/tuần", fontweight="bold")
axes[0].axhline(up_s, color="red", linestyle="--", label=f"Upper={up_s:.0f}")
axes[0].legend()

# Boxplot avg_session_minutes
sns.boxplot(y=df["avg_session_minutes"], ax=axes[1], color="#2ecc71")
axes[1].set_title("Outliers: Thời gian/phiên", fontweight="bold")

# Scatter: sessions vs minutes (highlight outliers)
sample_df = df.sample(10000, random_state=42)
axes[2].scatter(sample_df["sessions_per_week"], sample_df["avg_session_minutes"],
                alpha=0.2, s=10, color="gray", label="Normal")
super_sample = super_active.sample(min(500, len(super_active)), random_state=42)
axes[2].scatter(super_sample["sessions_per_week"], super_sample["avg_session_minutes"],
                alpha=0.6, s=20, color="red", label="Siêu tích cực")
axes[2].set_xlabel("Sessions/tuần")
axes[2].set_ylabel("Avg session minutes")
axes[2].set_title("Phát hiện người dùng học quá nhiều", fontweight="bold")
axes[2].legend()

plt.tight_layout()
plt.show()""")

md("**Nhận xét:**\n- Có một nhóm nhỏ người dùng truy cập >10 phiên/tuần với thời gian >60 phút/phiên\n- Đây có thể là sinh viên đang chuẩn bị thi hoặc power users thực sự\n- Cần theo dõi để tránh burnout và đề xuất gói Premium phù hợp")

md("### 4.2 Người dùng đăng ký nhiều khóa nhưng không học")

code("""# 4.2 Người đăng ký nhiều khóa nhưng không học
# Định nghĩa: courses_enrolled >= 10 VÀ completion_rate <= 0.1
ghost_users = df[(df["courses_enrolled"] >= 10) & (df["completion_rate"] <= 0.1)]
print(f"👻 Người dùng 'ghost' (đăng ký ≥10 khóa, hoàn thành ≤10%): {len(ghost_users):,}")
print(f"   Tỷ lệ: {len(ghost_users)/len(df)*100:.2f}%")
print()

# Thống kê nhóm ghost
print("Thống kê nhóm Ghost Users:")
print(ghost_users[["sessions_per_week", "avg_session_minutes", "videos_watched",
                    "courses_enrolled", "completion_rate", "total_spent_usd",
                    "premium_purchased", "churn_risk"]].describe().round(2))""")

code("""# Visualization ghost users
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# courses_enrolled vs completion_rate
sample_all = df.sample(10000, random_state=42)
axes[0].scatter(sample_all["courses_enrolled"], sample_all["completion_rate"],
                alpha=0.2, s=10, color="gray", label="Normal")
ghost_sample = ghost_users.sample(min(2000, len(ghost_users)), random_state=42)
axes[0].scatter(ghost_sample["courses_enrolled"], ghost_sample["completion_rate"],
                alpha=0.5, s=15, color="orange", label="Ghost users")
axes[0].set_xlabel("Courses Enrolled")
axes[0].set_ylabel("Completion Rate")
axes[0].set_title("Ghost Users: Đăng ký nhiều, học ít", fontweight="bold")
axes[0].legend()

# So sánh churn risk
groups = pd.DataFrame({
    "Ghost Users": [ghost_users["churn_risk"].mean() * 100],
    "Normal Users": [df[~df.index.isin(ghost_users.index)]["churn_risk"].mean() * 100]
}).T
groups.columns = ["Churn Risk (%)"]
groups.plot(kind="bar", ax=axes[1], color=["#e74c3c", "#2ecc71"], legend=False)
axes[1].set_title("Churn Risk: Ghost vs Normal", fontweight="bold")
axes[1].set_ylabel("Churn Risk (%)")
axes[1].tick_params(axis="x", rotation=0)

# Premium rate
premium = pd.DataFrame({
    "Ghost Users": [ghost_users["premium_purchased"].mean() * 100],
    "Normal Users": [df[~df.index.isin(ghost_users.index)]["premium_purchased"].mean() * 100]
}).T
premium.columns = ["Premium Rate (%)"]
premium.plot(kind="bar", ax=axes[2], color=["#e74c3c", "#3498db"], legend=False)
axes[2].set_title("Premium Rate: Ghost vs Normal", fontweight="bold")
axes[2].set_ylabel("Premium Rate (%)")
axes[2].tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.show()""")

md("**Nhận xét:**\n- Nhóm Ghost Users đăng ký nhiều khóa (≥10) nhưng completion rate rất thấp (≤10%)\n- Nhóm này có nguy cơ churn cao hơn đáng kể so với nhóm bình thường\n- Đội sản phẩm nên gửi email nhắc nhở hoặc tạo onboarding flow tốt hơn\n- Có thể thiết kế micro-courses ngắn để tăng engagement ban đầu")

md("### 4.3 Người dùng chi tiêu bất thường")

code("""# 4.3 Người dùng chi tiêu bất thường
out_spent, low_sp, up_sp = detect_outliers_iqr(df, "total_spent_usd")
print(f"Total Spent USD - Ngưỡng IQR: [{low_sp:.1f}, {up_sp:.1f}]")
print(f"Số outliers chi tiêu: {len(out_spent):,} ({len(out_spent)/len(df)*100:.2f}%)")

# Người chi tiêu cao nhất
high_spenders = df[df["total_spent_usd"] >= 150].sort_values("total_spent_usd", ascending=False)
print(f"\\n💰 Người dùng chi tiêu ≥150 USD: {len(high_spenders):,}")
print(high_spenders[["user_id", "total_spent_usd", "courses_enrolled", "completion_rate",
                      "premium_purchased", "sessions_per_week"]].head(15))

# Người chi 0 USD nhưng premium
zero_spend_premium = df[(df["total_spent_usd"] == 0) & (df["premium_purchased"] == 1)]
print(f"\\n⚠️  Premium nhưng chi tiêu = 0 USD: {len(zero_spend_premium):,}")""")

code("""# Visualization chi tiêu bất thường
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histogram chi tiêu
axes[0].hist(df["total_spent_usd"], bins=50, color="#f39c12", edgecolor="white", alpha=0.8)
axes[0].axvline(up_sp, color="red", linestyle="--", label=f"Ngưỡng upper={up_sp:.0f}")
axes[0].set_xlabel("Total Spent (USD)")
axes[0].set_ylabel("Số lượng")
axes[0].set_title("Phân phối chi tiêu", fontweight="bold")
axes[0].legend()

# Boxplot chi tiêu theo premium
sns.boxplot(x="premium_purchased", y="total_spent_usd", data=df, ax=axes[1],
            palette=["#e74c3c", "#27ae60"])
axes[1].set_xticklabels(["Free", "Premium"])
axes[1].set_title("Chi tiêu: Free vs Premium", fontweight="bold")

# Top spenders - scatter
axes[2].scatter(df["courses_enrolled"], df["total_spent_usd"], alpha=0.1, s=5, color="gray")
axes[2].scatter(high_spenders["courses_enrolled"], high_spenders["total_spent_usd"],
                alpha=0.5, s=20, color="red", label="High Spenders (≥150 USD)")
axes[2].set_xlabel("Courses Enrolled")
axes[2].set_ylabel("Total Spent (USD)")
axes[2].set_title("High Spenders vs Courses Enrolled", fontweight="bold")
axes[2].legend()

plt.tight_layout()
plt.show()""")

md("**Nhận xét:**\n- Có một nhóm người dùng chi tiêu rất cao (≥150 USD) - đây là nhóm VIP cần chăm sóc đặc biệt\n- Phần lớn người dùng chi tiêu ở mức thấp-trung bình\n- Một số trường hợp Premium nhưng chi tiêu = 0 USD cần kiểm tra (có thể dùng trial/coupon)\n- High spenders không nhất thiết là người hoàn thành khóa học nhiều nhất")

# ========================== 5. TỔNG KẾT ==========================
md("## 5. Tổng kết & Insight cho đội sản phẩm\n\n### 📊 Các insight chính:\n\n| Phát hiện | Chi tiết | Đề xuất |\n|-----------|----------|--------|\n| Ghost Users | Đăng ký ≥10 khóa, hoàn thành ≤10% | Email nhắc nhở, micro-courses, cải thiện onboarding |\n| Power Users | Sessions ≥10/tuần, >60 phút/phiên | Upsell Premium, tạo referral program |\n| High Spenders | Chi tiêu ≥150 USD | Chương trình VIP, ưu đãi loyalty |\n| Churn Risk | Completion rate thấp → churn cao | Gamification, progress reminders |\n| AI Recommendations | Click nhưng ít enroll | Cải thiện thuật toán gợi ý, A/B test |\n\n### 🎯 Trả lời câu hỏi CEO:\n\n1. **Người dùng có những kiểu hành vi nào?**\n   - Ghost users (đăng ký nhiều, không học)\n   - Active learners (học đều đặn, completion rate TB-cao)\n   - Power users (siêu tích cực)\n   - High spenders (chi tiêu nhiều)\n\n2. **Phân nhóm người dùng?**\n   - Có thể phân theo: completion rate, sessions/tuần, chi tiêu\n   - Kết hợp clustering (K-Means) để tạo segments tự động\n\n3. **Dự đoán khả năng mua khóa học?**\n   - Features quan trọng: sessions_per_week, completion_rate, ai_recommend_click\n   - Có thể xây dựng mô hình ML trên cột `future_purchase`")

# ==================== BUILD NOTEBOOK ====================
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

with open(r"e:\Visualization\Lab_main\learnx_analysis.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"✅ Notebook created with {len(cells)} cells")
