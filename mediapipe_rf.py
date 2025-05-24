import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 基于mediapipe提取手部特征并通过随机森林分类

CLASS_NAMES = [
    'A','B','C','D','E','F','G','H','I','K',
    'L','M','N','O','P','Q','R','S','T','U',
    'V','W','X','Y'
]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

def extract_hand_keypoints(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"跳过无效图像: {image_path}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    if not results.multi_hand_landmarks:
        return None
    keypoints = []
    for landmark in results.multi_hand_landmarks[0].landmark:
        keypoints.extend([landmark.x, landmark.y, landmark.z])
    return np.array(keypoints)

def extract_enhanced_features(image_path):
    kps = extract_hand_keypoints(image_path)
    if kps is None:
        return None
    wrist = kps[0:3]
    middle_mcp = kps[9*3:9*3+3]
    direction = middle_mcp - wrist
    direction = direction / np.linalg.norm(direction + 1e-6)  # 归一化
    return np.concatenate([kps, direction])

def load_or_process_features(data_root, save_path='processed_features.npz'):
    if os.path.exists(save_path):
        print("加载已有特征文件...")
        data = np.load(save_path)
        return data['X'], data['y']

    print("提取新特征...")
    features, labels = [], []
    train_dir = os.path.join(data_root, 'asl_alphabet_train', 'asl_alphabet_train')

    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(train_dir, class_name)
        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith(('.jpg', '.png')):
                continue
            img_path = os.path.join(class_dir, img_name)
            kps = extract_enhanced_features(img_path)
            if kps is not None:
                features.append(kps)
                labels.append(label_idx)

    X, y = np.array(features), np.array(labels)
    np.savez(save_path, X=X, y=y)
    return X, y

def main():
    data_root = "./data"
    X, y = load_or_process_features(data_root)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("训练中...")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
        n_jobs=-1,
        class_weight='balanced_subsample',
        random_state=42
    )
    clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"训练准确率: {train_acc:.4f}")
    print(f"测试准确率: {test_acc:.4f}")
    print("分类报告:")
    print(classification_report(y_test, test_pred, target_names=CLASS_NAMES))

    joblib.dump(clf, 'asl_classifier.pkl')
    print("模型保存完成")

if __name__ == "__main__":
    main()
