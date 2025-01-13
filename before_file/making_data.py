#%%
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

# 데이터 파일 경로 지정
file_path = 'data/ratings.dat'

# 데이터 불러오기: '::' 구분자로 데이터를 읽어옵니다.
columns = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv(file_path, sep='::', names=columns, engine='python')

# 1. 데이터 이진화: 5성급 평점은 1로, 나머지는 0으로 변환
df['rating'] = (df['rating'] == 5).astype(int)
df_rating_5 = df[df['rating'] == 1]

# 2. 고유한 아이템 목록에서 4819개의 아이템을 무작위로 선택 (중복 허용하지 않음)
np.random.seed(42)  # 시드 고정

# item 걸러내기 상호작용 횟수 10회 이하 버려
unique_item_index, item_counts = np.unique(df_rating_5['item_id'],return_counts = True)

len(item_counts[item_counts<=10])

filtered_item_indices = unique_item_index[item_counts > 10]
df_filtered = df_rating_5[df_rating_5['item_id'].isin(filtered_item_indices)]

# user 걸러내기
unique_user_index, user_counts = np.unique(df_filtered['user_id'],return_counts=True)
len(user_counts[user_counts<=10])
filtered_user_indices = unique_user_index[user_counts > 10]
df_filtered_total = df_filtered[df_filtered['user_id'].isin(filtered_user_indices)]

# 유저, 아이템 인덱스 재설정
user_id_mapping = {old_id: new_id for new_id, old_id in enumerate(df_filtered_total['user_id'].unique())}
df_filtered_total['user_id'] = df_filtered_total['user_id'].map(user_id_mapping)

# 4.2. 아이템 ID 재설정: 0부터 시작하는 새로운 인덱스로 매핑
item_id_mapping = {old_id: new_id for new_id, old_id in enumerate(df_filtered_total['item_id'].unique())}
df_filtered_total['item_id'] = df_filtered_total['item_id'].map(item_id_mapping)

#%%
# normal, skewed 데이터 나누기
item_popularity = df_filtered_total['item_id'].value_counts()
inverse_popularity = 1 / item_popularity
sampling_probabilities = inverse_popularity / inverse_popularity.sum()

# 5.2. 전체 상호작용의 40% 크기로 샘플링
np.random.seed(42)
num_samples = int(len(df_filtered_total) * 0.4)

# 비복원추출 방식으로 skewed 데이터 생성 (인기도 역수를 확률로 사용)
sampled_interactions = df_filtered_total.sample(
    n=num_samples,
    weights=df_filtered_total['item_id'].map(sampling_probabilities),
    replace=False
)

# 6. Normal 데이터 생성: Skewed 데이터에 포함되지 않은 나머지 데이터
normal_data = df_filtered_total.drop(sampled_interactions.index)

# 7. 결과 출력 및 확인
print(f"Skewed data size: {len(sampled_interactions)}")
print(f"Normal data size: {len(normal_data)}")

# Skewed 데이터셋의 아이템 빈도 분포 시각화
skewed_item_popularity = sampled_interactions['item_id'].value_counts()
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1,len(skewed_item_popularity)+1), skewed_item_popularity.values, marker='o', linestyle='-', color='skyblue')
plt.xlabel('Item ID')
plt.ylabel('Frequency')
plt.title('Skewed Data Item Frequency Distribution')
plt.grid(True)
plt.show()



# Normal 데이터셋의 아이템 빈도 분포 시각화
normal_item_popularity = normal_data['item_id'].value_counts()
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1,len(normal_item_popularity)+1), normal_item_popularity.values, marker='o', linestyle='-', color='palegreen')
plt.xlabel('Item ID')
plt.ylabel('Frequency')
plt.title('Normal Data Item Frequency Distribution')
plt.grid(True)
plt.show()

#%%
# 전체 데이터셋의 아이템 빈도 분포 시각화
plt.figure(figsize=(16, 10))  # 그래프 크기를 키움

# 그래프 라인과 마커 스타일 (선명한 보라색)
plt.plot(
    np.arange(1, len(total_item_popularity) + 1), 
    total_item_popularity.values, 
    marker='o', linestyle='-', linewidth=3, markersize=8, color='#6A5ACD', alpha=0.95
)

# 곡선 아래 영역을 은은한 보라색으로 음영 처리
plt.fill_between(
    np.arange(1, len(total_item_popularity) + 1), 
    total_item_popularity.values, 
    color='#B4A7D6', alpha=0.6
)

# 배경 색상 화이트로 설정
plt.gca().set_facecolor('white')

# 축 및 제목 스타일 (폰트 크기 확대)
plt.xlabel('Item ID', fontsize=20, fontweight='bold', color='#333333', labelpad=20)  # labelpad로 간격 조정
plt.ylabel('Frequency', fontsize=20, fontweight='bold', color='#333333', labelpad=20)
plt.title('Item Frequency Distribution', fontsize=28, fontweight='bold', color='#333333', pad=30)  # pad로 제목 간격 조정

# 그리드 스타일 (더 진하고 가시성 높은 점선)
plt.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.8, color='#A9A9A9')

# 축 레이블과 틱 스타일 (폰트 크기와 두께 조정)
plt.xticks(fontsize=16, color='#333333')  # x축 레이블 크기 키움
plt.yticks(fontsize=16, color='#333333')  # y축 레이블 크기 키움

# 범례 스타일
plt.legend(['Item Frequencies'], loc='upper right', fontsize=18, frameon=False)

# 축의 테두리 스타일 변경
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_color('#A9A9A9')
plt.gca().spines['bottom'].set_color('#A9A9A9')

# 이미지 저장
plt.tight_layout()
plt.savefig('item_frequency_distribution_large_font.png', dpi=300, bbox_inches='tight')

# 그래프 출력
plt.show()


# %%
