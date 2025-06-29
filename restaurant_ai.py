import cv2
from deepface import DeepFace
import tensorflow as tf
import requests
import os
from datetime import datetime
import time
from collections import Counter
import numpy as np
import db # db.py faylini import qilish

# --- YANGI FUNKSIYA (find_euclidean_distance) ---
# deepface.commons.distance.find_euclidean_distance o'rniga
def find_euclidean_distance(source_representation, test_representation):
    """
    Ikkita L2-normalizatsiya qilingan yuz embeddingi orasidagi Evklid masofasini hisoblaydi.
    """
    if not isinstance(source_representation, np.ndarray):
        source_representation = np.array(source_representation)
    if not isinstance(test_representation, np.ndarray):
        test_representation = np.array(test_representation)
    
    # Ikki vektor orasidagi Evklid masofasi
    distance = np.linalg.norm(source_representation - test_representation)
    return distance
# --- YANGI FUNKSIYA TUGADI ---


# --- Umumiy va TensorFlow sozlamalari ---
TF_LOG_LEVEL = 0 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(TF_LOG_LEVEL)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ TensorFlow GPU topildi: {gpus[0].name}")
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("✅ GPU xotirasining dinamik o'sishi yoqildi.")
    except RuntimeError as e:
        print(f"❌ GPU xotirasini sozlashda xato: {e}")
        pass 
    device_name = '/GPU:0'
else:
    print("❌ TensorFlow GPU topilmadi. CPUda ishga tushiriladi.")
    device_name = '/CPU:0'

print(f"Hozirda '{device_name}' qurilmasidan foydalanilmoqda.")

# --- Kamera sozlamalari ---
source_type = 'live_camera' 
camera_index = 0             
video_file_path = "sample_video.mp4" 
rtsp_url = "rtsp://admin:1q2w3e4r!@192.168.100.4:554/stream" 

# --- Yuzni aniqlovchi model fayllari ---
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
confidence_threshold = 0.8 
detect_only_first_face = False 

# --- Telegram sozlamalari ---
BOT_TOKEN = '7870745882:AAE-kUR_wHWKsm5HDHzcGEv4q7IihRuOL8Q' 
CHAT_ID = '-4680450156' 
TELEGRAM_SEND_INTERVAL = 30 

# --- Yuz tahlilini optimizatsiya qilish uchun sozlamalar ---
ANALYSIS_BUFFER_SIZE = 5 
analysis_history = [] 

# --- Mijozlarni qayta aniqlash (Re-identification) sozlamalari ---
EMBEDDING_MODEL_NAME = "Facenet" 
EMBEDDING_THRESHOLD = 0.8 # <-- Shu yerda 0.6 qiling (Evklid masofasi uchun!)
                           # Keyinchalik, loglardagi masofalarga qarab 0.7-1.0 gacha oshirish mumkin.

# --- Qayta aniqlash funksiyasi (restaurant_ai.py ichida qoladi) ---
def find_matching_customer(new_embedding, existing_customers_data, threshold):
    best_match_id = None
    min_distance = float('inf')

    print(f"DEBUG: Qidirilayotgan embedding. Uzunligi: {len(new_embedding)}") 
    
    if not existing_customers_data:
        print("DEBUG: Bazada mijozlar yo'q.") 
        return None, min_distance 

    for customer in existing_customers_data:
        # DeepFace'ning Evklid masofasi hisobini ishlatamiz (l2_normalize qilingan embeddinglar uchun)
        distance = find_euclidean_distance(new_embedding, customer['embedding']) 

        print(f"DEBUG: Mijoz ID {customer['id']} bilan masofa: {distance:.4f}") 
        
        if distance < min_distance:
            min_distance = distance
            best_match_id = customer['id']
    
    print(f"DEBUG: Eng yaqin masofa: {min_distance:.4f}, Eng yaqin ID: {best_match_id}") 
    if min_distance <= threshold: # Chegaradan kichik yoki teng bo'lsa, mos keladi
        print(f"DEBUG: Mos kelish topildi! ID: {best_match_id}, Masofa: {min_distance:.4f} <= Chegara: {threshold:.4f}")
        return best_match_id, min_distance
    print(f"DEBUG: Mos kelish topilmadi. Eng yaqin masofa: {min_distance:.4f} > Chegara: {threshold:.4f}") 
    return None, min_distance 


# --- Ma'lumotlar bazasini ishga tushirish (db.py orqali) ---
db.create_tables() 


# --- Model fayllarining mavjudligini tekshirish ---
if not os.path.exists(modelFile) or not os.path.exists(configFile):
    print(f"❌ Yuzni aniqlash modeli fayllari topilmadi: {modelFile} yoki {configFile}")
    print("Iltimas, ushbu fayllarni kod bilan bir papkaga joylashtiring.")
    exit()

try:
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    print("✅ Yuzni aniqlovchi model muvaffaqiyatli yuklandi.")
except Exception as e:
    print(f"❌ Yuzni aniqlovchi modelni yuklashda xatolik yuz berdi: {e}")
    exit()

# --- Kamera/Video manbasini ishga tushurish ---
cap = None
if source_type == 'live_camera':
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW) 
    print(f"Kamera indeksi {camera_index} ishga tushirilmoqda.")
elif source_type == 'video_file':
    if not os.path.exists(video_file_path):
        print(f"❌ Video fayl topilmadi: {video_file_path}")
        print("Iltimas, test uchun video faylni loyiha papkasiga joylashtiring.")
        exit()
    cap = cv2.VideoCapture(video_file_path)
    print(f"Video fayl '{video_file_path}' ishga tushirilmoqda.")
elif source_type == 'rtsp':
    cap = cv2.VideoCapture(rtsp_url)
    print(f"RTSP oqimi '{rtsp_url}' ishga tushirilmoqda.")
else:
    print("❌ Noto'g'ri 'source_type' qiymati. 'live_camera', 'video_file' yoki 'rtsp' bo'lishi kerak.")
    exit()

if not cap.isOpened():
    print("❌ Kamera/Video manbasi ochilmadi! Manba mavjudligini va sozlamalarni tekshiring.")
    exit()
print("📷 Kamera/Video manbasi ishga tushdi.")

# --- Boshqa o'zgaruvchilar ---
last_sent_time = 0  
last_customer_id = None 
customer_send_interval = 60 
last_customer_sent_time = {} 

# --- Asosiy sikl ---
try:
    with tf.device(device_name): 
        while True:
            ret, frame = cap.read()
            if not ret:
                print("🛑 Ramka o'qilmadi. Manba tugadi yoki ulanishda muammo.")
                if source_type == 'video_file':
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read() 
                    if not ret: 
                        print("❌ Videoni boshidan qayta o'qib bo'lmadi. Tugatildi.")
                        break
                else: 
                    time.sleep(1) 
                    cap.release() 
                    if source_type == 'rtsp':
                        cap = cv2.VideoCapture(rtsp_url)
                    elif source_type == 'live_camera':
                        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW) 
                    
                    if not cap.isOpened():
                        print("❌ Manbaga qayta ulanib bo'lmadi. Dastur tugatildi.")
                        break
                    continue 

            # --- Debugging ---
            print(f"DEBUG: Ramka {time.time():.2f} vaqtida qayta ishlanmoqda...") 
            # --- Debugging tugadi ---

            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
            net.setInput(blob)
            
            start_time = time.time()
            detections = net.forward() 
            detection_time = time.time() - start_time

            detected_faces_count = 0
            current_frame_analysis = [] 
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > confidence_threshold: 
                    detected_faces_count += 1
                    box = detections[0, 0, i, 3:7] * [w, h, w, h]
                    (x1, y1, x2, y2) = box.astype("int")

                    face = frame[y1:y2, x1:x2]
                    if face.shape[0] == 0 or face.shape[1] == 0:
                        continue 

                    # --- Kayfiyat, Demografiya (Jins, Yosh) va Embedding aniqlash ---
                    try:
                        results_analysis = DeepFace.analyze(
                            img_path=face,
                            actions=["emotion", "gender", "age"], 
                            detector_backend="retinaface", 
                            enforce_detection=False 
                        )
                        emotion = results_analysis[0]["dominant_emotion"]
                        gender = results_analysis[0]["dominant_gender"] 
                        age = int(results_analysis[0]["age"]) 

                        results_represent = DeepFace.represent(
                            img_path=face,
                            model_name=EMBEDDING_MODEL_NAME, 
                            detector_backend="retinaface",
                            enforce_detection=False
                        )
                        face_embedding = results_represent[0]["embedding"]
                        
                        # Embeddingni L2-normallashtiramiz (restaurant_ai.py ichida)
                        if isinstance(face_embedding, list): 
                            face_embedding = np.array(face_embedding)
                        if len(face_embedding) > 0: 
                            face_embedding = db.l2_normalize(face_embedding) # db.l2_normalize() chaqiriladi
                        else:
                            print(f"❌ Xatolik: DeepFace.represent bo'sh embedding qaytardi. O'tkazib yuborildi.")
                            continue 
                        
                        current_frame_analysis.append({"age": age, "gender": gender, "emotion": emotion, "embedding": face_embedding, "face_image": face}) 

                        # --- Vizualizatsiya va ma'lumotni kadrga yozish (har bir yuz uchun) ---
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                        cv2.putText(frame, emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(frame, f"J: {gender}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(frame, f"Y: {age}", (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        if detect_only_first_face: 
                            break 

                    except Exception as e:
                        print(f"❌ Yuzni tahlil qilishda xatolik (DeepFace xatosi): {e}") 
                        pass 

            # --- Kadr bo'yicha umumiy demografik ma'lumotni hisoblash va Telegramga yuborish ---
            if current_frame_analysis: 
                
                analysis_history.extend(current_frame_analysis)
                if len(analysis_history) > ANALYSIS_BUFFER_SIZE:
                    analysis_history = analysis_history[-ANALYSIS_BUFFER_SIZE:] 

                current_time = time.time()
                
                if current_time - last_sent_time >= TELEGRAM_SEND_INTERVAL:
                    
                    if analysis_history: 
                        ages_in_buffer = [item['age'] for item in analysis_history if isinstance(item['age'], int)]
                        genders_in_buffer = [item['gender'] for item in analysis_history if item['gender'] != "unknown"]
                        emotions_in_buffer = [item['emotion'] for item in analysis_history if item['emotion'] != "unknown"]
                        
                        embeddings_in_buffer = [item['embedding'] for item in analysis_history if 'embedding' in item and item['embedding'] is not None]

                        avg_age_val = int(sum(ages_in_buffer) / len(ages_in_buffer)) if ages_in_buffer else "N/A"
                        dominant_gender_val = Counter(genders_in_buffer).most_common(1)[0][0] if genders_in_buffer else "N/A"
                        dominant_emotion_val = Counter(emotions_in_buffer).most_common(1)[0][0] if emotions_in_buffer else "N/A"
                        
                        customer_id = None
                        customer_status_message = "Mijoz aniqlanmadi." 
                        profile_msg = "Mijoz profili topilmasi." 
                        
                        if embeddings_in_buffer: 
                            latest_embedding = embeddings_in_buffer[-1] 
                            
                            existing_customers_data = db.get_all_customer_embeddings()
                            
                            print(f"\nDEBUG: Mijozni aniqlashga urinish...")
                            print(f"DEBUG: Bazada {len(existing_customers_data)} ta mijoz mavjud.")

                            match_id, min_dist = find_matching_customer(latest_embedding, existing_customers_data, EMBEDDING_THRESHOLD) 
                            
                            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                            # --- Profil rasmini saqlash ---
                            profile_image_path = None
                            face_image_to_save = None

                            if current_frame_analysis and 'face_image' in current_frame_analysis[-1]:
                                face_image_to_save = current_frame_analysis[-1]['face_image']
                            
                            if face_image_to_save is not None:
                                profile_base_dir = "customer_faces"
                                os.makedirs(profile_base_dir, exist_ok=True) 

                                profile_dir_date = os.path.join(profile_base_dir, datetime.now().strftime("%Y%m%d")) 
                                os.makedirs(profile_dir_date, exist_ok=True) 
                                
                                profile_image_filename = f"profile_{timestamp_str.replace(' ', '_').replace(':', '-')}.jpg"
                                profile_image_path = os.path.join(profile_dir_date, profile_image_filename)
                                try:
                                    cv2.imwrite(profile_image_path, face_image_to_save)
                                    print(f"📸 Profil rasmi saqlandi: {profile_image_path}")
                                except Exception as e:
                                    print(f"❌ Profil rasmini saqlashda xatolik: {e}")
                                    profile_image_path = None 
                            # --- Rasm saqlash tugadi ---

                            if match_id:
                                customer_id = match_id
                                customer_profile, customer_visits, customer_purchases = db.get_customer_full_profile(customer_id)
                                customer_status_message = f"Qayta kelgan mijoz (ID: {customer_id}, o'xshashlik: {min_dist:.2f})"
                                db.update_customer_profile(customer_id, dominant_gender_val, avg_age_val, timestamp_str, customer_profile['total_visits'] + 1)
                                
                            else:
                                customer_id = db.add_customer(latest_embedding, dominant_emotion_val, dominant_gender_val, avg_age_val, timestamp_str, name="Noma'lum", profile_image_path=profile_image_path) 
                                customer_status_message = f"Yangi mijoz (ID: {customer_id})"
                            
                            db.log_visit(customer_id, timestamp_str, dominant_emotion_val, dominant_gender_val, avg_age_val)
                            
                            customer_profile, customer_visits, customer_purchases = db.get_customer_full_profile(customer_id)

                            profile_msg = f"Ism: {customer_profile['name']}\n" \
                                          f"Umumiy tashrif: {customer_profile['total_visits']}\n" \
                                          f"Jami sarf: ${customer_profile['total_amount_spent']:.2f}\n" \
                                          f"Cashback ballari: {customer_profile['cashback_points']:.2f}\n" \
                                          f"Birinchi tashrif: {customer_profile['first_visit_time']}\n" \
                                          f"Oxirgi tashrif: {customer_profile['last_visit_time']}"

                            if customer_profile['total_visits'] % 5 == 0 and customer_profile['total_visits'] > 0:
                                dummy_products = [{"name": "Kofe", "qty": 1}, {"name": "Burger", "qty": 1}]
                                dummy_amount = np.random.uniform(5.0, 25.0) 
                                db.record_purchase(customer_id, timestamp_str, dummy_amount, dummy_products, "Karta") 
                                profile_msg += f"\n💰 Yangi xarid qayd etildi: ${dummy_amount:.2f}"
                                if customer_profile['total_visits'] % 10 == 0: 
                                    profile_msg += "\n🎉 Bonus chegirma: 10%!" 
                        
                            last_sent_time = current_time 
                            try:
                                message = (f"📈 {customer_status_message}\n"
                                        f"👥 Aniqlangan yuzlar (bu kadrda): {detected_faces_count}\n"
                                        f"🧠 Asosiy kayfiyat: {dominant_emotion_val}\n"
                                        f"🚻 Asosiy jins: {dominant_gender_val}\n"
                                        f"🎂 O'rtacha yosh: {avg_age_val}\n\n"
                                        f"--- Mijoz profili ---\n{profile_msg}")
                                
                                requests.post(
                                    f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                                    data={"chat_id": CHAT_ID, "text": message},
                                    timeout=5 
                                )
                                
                                if profile_image_path and os.path.exists(profile_image_path): 
                                    with open(profile_image_path, "rb") as photo:
                                        requests.post(
                                            f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
                                            data={"chat_id": CHAT_ID},
                                            files={"photo": photo},
                                            timeout=10 
                                        )
                                    print("✅ Telegramga profil rasmi yuborildi.")
                                else:
                                    print("❌ Telegramga profil rasmi yuborilmadi: rasm fayli topilmadi.")
                            except Exception as e:
                                print(f"❌ Telegram yuborishda xatolik: {e}")
                        else:
                            print("🚫 Buferda ma'lumot yo'q, Telegramga yuborilmaydi (shartlar bajarilmadi).")
                else:
                    print(f"⏳ Juda tez yuborilmoqda — keyingi {TELEGRAM_SEND_INTERVAL} soniyada.")
            else:
                pass 

            cv2.putText(frame, f"Faces: {detected_faces_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Detection Time: {detection_time:.2f}s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Face Detection and Emotion Analysis", frame) 
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("🛑 To‘xtatildi: 'q' tugmasi.")
                break

except KeyboardInterrupt:
    print("🛑 Jarayon Ctrl+C orqali to‘xtatildi.")
except Exception as e:
    print(f"❌ Kutilmagan global xatolik yuz berdi: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Manba resurslari ozod qilindi.")