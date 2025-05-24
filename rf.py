import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 效果不好，分不清人脸和手

CLASS_NAMES = [
    'A','B','C','D','E','F','G','H','I','K',
    'L','M','N','O','P','Q','R','S','T','U',
    'V','W','X','Y'
]

IMG_SIZE = (64, 64)  # 图像缩放尺寸

def extract_image_features(image_path):
    """不使用 MediaPipe，直接提取图像灰度值作为特征"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"跳过无效图像: {image_path}")
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    normalized = blurred.astype('float32') / 255.0
    return normalized.flatten()  # 展平成一维向量

def load_or_process_features(data_root, save_path='processed_image_features.npz'):
    if os.path.exists(save_path):
        print("加载已有图像特征文件...")
        data = np.load(save_path)
        return data['X'], data['y']

    print("开始提取图像特征...")
    features, labels = [], []
    train_dir = os.path.join(data_root, 'asl_alphabet_train', 'asl_alphabet_train')

    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(train_dir, class_name)
        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith(('.jpg', '.png')):
                continue
            img_path = os.path.join(class_dir, img_name)
            feat = extract_image_features(img_path)
            if feat is not None:
                features.append(feat)
                labels.append(label_idx)

    X, y = np.array(features), np.array(labels)
    np.savez(save_path, X=X, y=y)
    return X, y

def main():
    data_root = r"E:\360MoveData\Users\HYT\Desktop\AI_Interactive_Practice\Project\data"
    X, y = load_or_process_features(data_root)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("训练随机森林模型中...")
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
    print("模型保存为 asl_classifier.pkl")

if __name__ == "__main__":
    main()
