applicant_data_expanded =  pd.read_csv(r'C:\Users\leezoo\Desktop\졸업작품\graduation_inu\dummy\applicant_data_10000.csv')
team_leader_data_expanded =  pd.read_csv(r'C:\Users\leezoo\Desktop\졸업작품\graduation_inu\dummy\team_leader_data_10000.csv')
# 지원자와 팀 리더 데이터를 하나의 DataFrame으로 결합
combined_data = pd.concat([applicant_data_expanded, team_leader_data_expanded], ignore_index=True)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import seaborn as sns
from gensim.models import Word2Vec

# 데이터 로드 및 전처리 (이전과 동일)
data = combined_data
data = data.set_index('ID')

# Word2Vec 모델 훈련 및 데이터 벡터화 (이전과 동일)
def prepare_and_train_word2vec(data, tech_col, prog_col):
    combined_skills = data[tech_col] + ', ' + data[prog_col]
    skills_list = combined_skills.apply(lambda x: x.split(', '))
    model = Word2Vec(sentences=skills_list, vector_size=100, window=5, min_count=1, workers=4)
    return model, skills_list

data_model, data_skills = prepare_and_train_word2vec(data, 'Technical_Skills', 'Programming_Languages')
data_vectors = np.array([np.mean([data_model.wv[word] for word in skill if word in data_model.wv], axis=0) for skill in data_skills])

# 전처리 및 피처 결합 (이전과 동일)
numeric_features_data = ['Experience/Team_Size']
categorical_features_data = [col for col in data.columns if data[col].dtype == 'object' and col not in ['Technical_Skills', 'Programming_Languages']]

def setup_and_apply_preprocessor(data, numeric_features, categorical_features):
    preprocessor = ColumnTransformer([
        ('num', MinMaxScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])
    return preprocessor.fit_transform(data)

data_processed = setup_and_apply_preprocessor(data, numeric_features_data, categorical_features_data)
data_combined_features = combine_features(data_vectors, data_processed.toarray())

# PCA를 적용하여 2D로 변환
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_combined_features)

# PCA 결과에 대해 K-means 클러스터링 수행
kmeans = KMeans(n_clusters=4, random_state=0).fit(data_pca)
labels = kmeans.labels_
# 각 클러스터에 속한 샘플 ID를 저장 및 출력
cluster_ids = {}
for i in range(len(np.unique(labels))):
    cluster_ids[i] = data.index[labels == i].tolist()  # DataFrame의 인덱스를 사용
    print(f"Cluster {i} sample IDs:\n{cluster_ids[i]}\n")

centers_pca = kmeans.cluster_centers_

# PCA 결과 시각화
plt.figure(figsize=(10, 7))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='rainbow', alpha=0.7)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], marker='x', s=200, c='black')
plt.title('K-means Clustering with PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# 클러스터 내 매칭 구현
def match_applicants_to_leaders(cluster_ids, data):
    matches = {}
    for cluster, ids in cluster_ids.items():
        # 클러스터 내에서 지원자와 팀 리더 분리
        applicants = [id for id in ids if id.startswith('A')]
        leaders = [id for id in ids if id.startswith('T')]
        
        # 가능한 경우, 각 지원자를 한 팀 리더와 매칭
        paired = list(zip(applicants, leaders))  # 짧은 쪽에 맞춰서 쌍을 만듦
        matches[cluster] = paired
    
    return matches

# 매칭 결과 계산
matches = match_applicants_to_leaders(cluster_ids, data)

# 매칭 결과 출력
for cluster, match_list in matches.items():
    print(f"Cluster {cluster} matches:")
    for applicant, leader in match_list:
        print(f"  Applicant {applicant} is matched with Leader {leader}")
    print("\n")

