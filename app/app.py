import os
import json
import joblib
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from scipy.signal import butter, filtfilt, find_peaks
from flask import Flask, request, jsonify
import traceback
import warnings
import uuid
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# 忽略警告
warnings.filterwarnings('ignore', category=RuntimeWarning)

# 初始化Flask应用
app = Flask(__name__)

# 添加限流器，限制每个IP地址每分钟最多访问5次
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["5 per minute"],
    storage_uri="memory://",
)

# 从环境变量读取模型路径，这是Docker中的最佳实践
MODEL_DIR = os.environ.get('MODEL_DIR', '/app/models')
SBP_MODEL_PATH = os.path.join(MODEL_DIR, 'sbp_model.joblib')
DBP_MODEL_PATH = os.path.join(MODEL_DIR, 'dbp_model.joblib')
SPO2_PARAMS_PATH = os.path.join(MODEL_DIR, 'spo2_params.json')

# 确保上传文件夹存在
UPLOAD_FOLDER = '/tmp/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 全局变量用于存放加载的模型
sbp_model = None
dbp_model = None
spo2_params = None
face_mesh = None

# 加载模型的函数
def load_models():
    global sbp_model, dbp_model, spo2_params, face_mesh
    
    try:
        sbp_model = joblib.load(SBP_MODEL_PATH)
        dbp_model = joblib.load(DBP_MODEL_PATH)
        
        with open(SPO2_PARAMS_PATH, 'r') as f:
            spo2_params = json.load(f)
        
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False, 
            max_num_faces=1, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        
        print("所有模型和依赖加载成功！")
        return True
    except Exception as e:
        print(f"致命错误：模型加载失败: {e}\n{traceback.format_exc()}")
        return False

# 核心算法函数（这是我们之前所有研发工作的结晶）
def calculate_vitals_from_video(video_path, fps_default=30.0):
    try:
        # --- 1. 信号提取 ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None, "无法打开视频文件"
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps == 0 or fps > 60: fps = fps_default
        
        FOREHEAD_LANDMARKS = [103, 104, 67, 10, 336, 297, 332, 333]
        r_raw, g_raw, b_raw = [], [], []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                forehead_points = [(int(face_landmarks.landmark[i].x * frame.shape[1]), int(face_landmarks.landmark[i].y * frame.shape[0])) for i in FOREHEAD_LANDMARKS]
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(forehead_points)], 255)
                mean_val = cv2.mean(frame, mask=mask)
                r_raw.append(mean_val[2]); g_raw.append(mean_val[1]); b_raw.append(mean_val[0])
        cap.release()
        
        if len(g_raw) < fps * 5: return None, "视频时长过短或未能持续检测到人脸"

        # --- 2. 信号处理 ---
        fs = fps
        low_cutoff, high_cutoff = 0.7, 3.0
        nyquist = 0.5 * fs
        b, a = butter(4, [low_cutoff / nyquist, high_cutoff / nyquist], btype='band')
        g_processed = filtfilt(b, a, g_raw)
        r_filtered = filtfilt(b, a, r_raw)
        b_filtered = filtfilt(b, a, b_raw)

        # --- 3. 计算心率 (HR) ---
        valleys, _ = find_peaks(-g_processed, height=0.1, distance=fs/3.0)
        if len(valleys) < 5: return None, "有效心跳信号过少"
        hr = (len(valleys) / (len(g_processed) / fs)) * 60

        # --- 4. 提取血压 (BP) 全部特征 ---
        all_beats_features = []
        for i in range(len(valleys) - 1):
            start_sample, end_sample = valleys[i], valleys[i+1]
            ppg_beat = g_processed[start_sample:end_sample]
            time_beat = np.arange(len(ppg_beat)) / fs
            
            peak_indices, _ = find_peaks(ppg_beat)
            if len(peak_indices) > 0:
                systolic_peak_index = peak_indices[np.argmax(ppg_beat[peak_indices])]
                pulse_amplitude = ppg_beat[systolic_peak_index] - ppg_beat[0]
                systolic_rise_time = time_beat[systolic_peak_index]
                half_height = ppg_beat[0] + 0.5 * pulse_amplitude
                above_half_indices = np.where(ppg_beat > half_height)[0]
                pulse_width_50 = (above_half_indices[-1] - above_half_indices[0]) / fs if len(above_half_indices) > 1 else np.nan
                
                vpg_beat = np.diff(ppg_beat); apg_beat = np.diff(vpg_beat)
                apg_peaks, _ = find_peaks(apg_beat); apg_valleys, _ = find_peaks(-apg_beat)
                a_wave_h, b_wave_h, b_over_a = np.nan, np.nan, np.nan
                if len(apg_peaks) > 0 and len(apg_valleys) > 0:
                    a_wave_idx = apg_peaks[0]
                    possible_b_waves = apg_valleys[apg_valleys > a_wave_idx]
                    if len(possible_b_waves) > 0:
                        b_wave_idx = possible_b_waves[0]
                        a_wave_h = apg_beat[a_wave_idx]; b_wave_h = apg_beat[b_wave_idx]
                        if a_wave_h != 0: b_over_a = b_wave_h / a_wave_h
                
                all_beats_features.append({
                    'pulse_amplitude': pulse_amplitude, 'systolic_rise_time': systolic_rise_time, 'pulse_width_50': pulse_width_50,
                    'a_wave_height': a_wave_h, 'b_wave_height': b_wave_h, 'b_over_a_ratio': b_over_a
                })
        
        if not all_beats_features or len(all_beats_features) < 5: return None, "未能提取到足够数量的有效心跳波形"
        features_df = pd.DataFrame(all_beats_features).dropna()
        if len(features_df) < 5: return None, "有效心跳波形质量不佳"

        # --- 5. 特征聚合与分离 ---
        final_features_dbp_dict = {f'mean_{col}': features_df[col].mean() for col in features_df.columns}
        final_features_dbp_dict.update({f'std_{col}': features_df[col].std() for col in features_df.columns})
        df_for_dbp = pd.DataFrame([final_features_dbp_dict])
        
        sbp_feature_names = ['mean_pulse_amplitude', 'std_pulse_amplitude', 'mean_systolic_rise_time', 
                             'std_systolic_rise_time', 'mean_pulse_width_50', 'std_pulse_width_50']
        df_for_sbp = df_for_dbp[sbp_feature_names]
        
        # --- 6. 提取血氧 (SpO2) 特征 ---
        dc_r, dc_b = np.mean(r_raw), np.mean(b_raw)
        ac_r, ac_b = np.sqrt(np.mean(np.square(r_filtered))), np.sqrt(np.mean(np.square(b_filtered)))
        spo2_ratio = (ac_r / dc_r) / (ac_b / dc_b) if dc_r > 0 and dc_b > 0 else 0
        
        # --- 7. 进行预测 ---
        sbp_pred = sbp_model.predict(df_for_sbp)[0]
        dbp_pred = dbp_model.predict(df_for_dbp)[0]
        spo2_pred = spo2_params['A'] - spo2_params['B'] * spo2_ratio

        results = {
            "heart_rate": round(hr, 1),
            "sbp": round(float(sbp_pred), 1),
            "dbp": round(float(dbp_pred), 1),
            "spo2": round(spo2_pred, 1)
        }
        return results, "成功"

    except Exception as e:
        traceback.print_exc()
        return None, f"算法处理时发生未知错误: {e}"


# API端点
@app.route('/predict', methods=['POST'])
@limiter.limit("5 per minute") # 应用速率限制
def predict():
    # 简单的token验证函数（实际应用中应替换为更安全的验证）
    # token = request.headers.get('X-Mini-Token')
    # if token != "SECURE_TOKEN_123":
    #     return jsonify({"status": "error", "message": "无效或缺少token"}), 401
    
    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "请求中未找到视频文件"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"status": "error", "message": "未选择视频文件"}), 400
        
    if file:
        temp_filename = f"{uuid.uuid4()}.mp4"
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(video_path)
        
        results, message = calculate_vitals_from_video(video_path)
        
        os.remove(video_path)
        
        if results:
            return jsonify({"status": "success", "data": results})
        else:
            return jsonify({"status": "error", "message": message}), 500

# 在应用启动时加载模型
if not load_models():
    print("服务启动失败：模型加载错误，请检查模型文件路径是否正确。")
    # 在生产环境中，可以选择退出或让其在后续请求中报错
    # exit(1)

# Gunicorn作为入口时，不会执行 __name__ == '__main__' 下的代码
# 这部分仅用于本地直接运行 python app.py 进行调试
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)