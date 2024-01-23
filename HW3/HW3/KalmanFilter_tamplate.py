import numpy as np

# 初始狀態估計
x_est = np.array([[0]])  # 初始位置
P_est = np.array([[1]])  # 初始估計不確定性

# 系統的動態模型參數
A = np.array([[1]])  # 狀態轉移矩陣
B = np.array([[1]])  # 控制輸入矩陣
C = np.array([[0.0001]])  # 過程噪聲協方差

# 觀測模型參數
H = np.array([[1]])  # 觀測矩陣
R = np.array([[0.1]])  # 觀測噪聲協方差

def kalman_filter(x_est, P_est, control, measurement):
    """
    執行卡爾曼濾波的一個迭代。
    
    Args:
        x_est: 上一時刻的狀態估計
        P_est: 上一時刻的估計不確定性
        control: 控制變量
        measurement: 當前時刻的觀測值
    
    Returns:
        更新後的狀態估計和估計不確定性
    """
    # 預測步驟
    x_pred = A @ x_est + B @ np.array([control])
    P_pred = A @ P_est @ A.T + C

    # 更新步驟
    y = measurement - H @ x_pred  # 觀測殘差
    S = H @ P_pred @ H.T + R  # 殘差協方差
    K = P_pred @ H.T @ np.linalg.inv(S)  # 卡爾曼增益
    x_est_new = x_pred + K @ y
    P_est_new = P_pred - K @ H @ P_pred

    return x_est_new, P_est_new

# 測試卡爾曼濾波器
measurements = [1, 2, 3, 4, 5]  # 一些模擬的測量值
controls = [0.1, 0.1, 0.1, 0.1, 0.1]  # 一些模擬的控制輸入
for control, measurement in zip(controls, measurements):
    x_est, P_est = kalman_filter(x_est, P_est, control, measurement)
    print(f"Updated State Estimate: {x_est}")