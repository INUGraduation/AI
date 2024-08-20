from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn.cluster import KMeans

app = Flask(__name__)

# 머신러닝 모델, 전처리 파이프라인, 클러스터 모델 로드
with open('model_pipeline.pkl', 'rb') as file:
    preprocessor, kmeans_model = pickle.load(file)

@app.route('/recommend', methods=['POST'])
def recommend():
    # JSON 데이터 수신
    input_json = request.get_json()
    client_data = pd.DataFrame([input_json])  # JSON 데이터를 데이터프레임으로 변환

    # 데이터 전처리
    processed_data = preprocessor.transform(client_data)

    # 클라이언트가 속할 클러스터 예측
    client_cluster = kmeans_model.predict(processed_data)

    # 같은 클러스터에 속한 팀 리더 찾기 (여기서는 예시 데이터 사용)
    # 실제 구현에서는 팀 리더 데이터와 클러스터 매핑이 필요
    team_leaders = ['T1', 'T2', 'T3']  # 이는 예시 데이터로, 실제 데이터에 맞게 조정 필요

    # 결과를 JSON으로 반환
    return jsonify({'client_id': input_json['ID'], 'recommended_leaders': team_leaders})

if __name__ == '__main__':
    app.run(debug=True)
