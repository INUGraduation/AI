import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# 데이터 로드 및 전처리
data = pd.read_csv(r'C:\Users\leezoo\Desktop\졸업작품\graduation_inu\dummy\combined_data_100.csv')
data = data.set_index('ID')

# 단어 사전 구축 및 단어 인덱스 매핑
def build_vocab(skills_list):
    word_to_ix = {}
    for skills in skills_list:
        for word in skills:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    return word_to_ix

# Word2Vec 스타일의 임베딩을 PyTorch로 구현
class SkillEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkillEmbeddingModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, inputs):
        return self.embeddings(inputs)

# 데이터를 PyTorch 임베딩 모델로 전환 및 학습
def prepare_and_train_embedding_pytorch(data, tech_col, prog_col, embedding_dim=100, epochs=100, lr=0.01):
    combined_skills = data[tech_col] + ', ' + data[prog_col]
    skills_list = combined_skills.apply(lambda x: x.split(', '))
    word_to_ix = build_vocab(skills_list)
    vocab_size = len(word_to_ix)
    
    # 모델 초기화
    model = SkillEmbeddingModel(vocab_size, embedding_dim)
    loss_function = nn.MSELoss()  # 손실 함수 (임의로 MSELoss 사용, 개선 가능)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # 학습 데이터 준비
    for epoch in range(epochs):
        total_loss = 0
        for skills in skills_list:
            skill_indices = torch.tensor([word_to_ix[word] for word in skills], dtype=torch.long)
            skill_vectors = model(skill_indices)
            loss = loss_function(skill_vectors.mean(dim=0), torch.zeros(embedding_dim))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss}")

    return model, word_to_ix, skills_list

# PyTorch로 Word2Vec 학습 및 데이터 벡터화
embedding_dim = 100
embedding_model, word_to_ix, data_skills = prepare_and_train_embedding_pytorch(
    data, 'Technical_Skills', 'Programming_Languages', embedding_dim=embedding_dim)

def get_skill_vectors_pytorch(model, word_to_ix, skills_list):
    vectors = []
    for skills in skills_list:
        skill_indices = torch.tensor([word_to_ix[word] for word in skills], dtype=torch.long)
        skill_vector = model(skill_indices).mean(dim=0).detach().numpy()
        vectors.append(skill_vector)
    return np.array(vectors)

data_vectors = get_skill_vectors_pytorch(embedding_model, word_to_ix, data_skills)

# 전처리 및 피처 결합 
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
