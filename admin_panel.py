import streamlit as st
import db
import os
import cv2
import numpy as np
from datetime import datetime
import json
import insightface
import onnxruntime
import sys

# --- Admin Panel sozlamalari ---
CUSTOMER_PROFILE_DIR = "customer_faces"
MODEL_ROOT = "./models"

if not os.path.exists(CUSTOMER_PROFILE_DIR):
    os.makedirs(CUSTOMER_PROFILE_DIR)
if not os.path.exists(MODEL_ROOT):
    os.makedirs(MODEL_ROOT)

st.set_page_config(layout="wide", page_title="Aqlli Restoran Admin Paneli")
st.title("👨‍🍳 Aqlli Restoran Admin Paneli")

try:
    db.create_tables()
except Exception as e:
    st.error(f"❌ Ma'lumotlar bazasi jadvallarini yaratishda xatolik: {e}")
    st.stop()

@st.cache_resource
def load_insightface_model():
    try:
        app = insightface.app.FaceAnalysis(name='buffalo_l', root=MODEL_ROOT, providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        return app
    except Exception as e:
        st.error(f"InsightFace modelini yuklashda xatolik: {e}")
        st.stop()

face_analyser = load_insightface_model()
st.sidebar.info("✅ InsightFace modeli yuklandi.")

def display_image_from_path(path):
    if path and os.path.exists(path):
        try:
            img_bgr = cv2.imread(path)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                return img_rgb
            else:
                return None
        except Exception:
            return None
    return None

def refresh_data():
    st.session_state.last_refresh_time = datetime.now()
    st.rerun()

# --- Session State ni boshlash ---
if 'current_customer_id' not in st.session_state:
    st.session_state.current_customer_id = None
if 'current_customer_name' not in st.session_state:
    st.session_state.current_customer_name = ""
if 'last_refresh_time' not in st.session_state:
    st.session_state.last_refresh_time = datetime.now()
if 'detected_customer_id' not in st.session_state:
    st.session_state.detected_customer_id = None
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False
if 'add_new_customer_prompt' not in st.session_state:
    st.session_state.add_new_customer_prompt = False
if 'new_customer_embedding_temp' not in st.session_state:
    st.session_state.new_customer_embedding_temp = None
if 'new_customer_image_temp' not in st.session_state:
    st.session_state.new_customer_image_temp = None
if 'last_unknown_detection_time' not in st.session_state: # Takroriy qo'shilishni oldini olish uchun
    st.session_state.last_unknown_detection_time = datetime.min
if 'unknown_detection_cooldown' not in st.session_state: # Takroriy qo'shilishni oldini olish uchun
    st.session_state.unknown_detection_cooldown = 10 # 10 soniya
if 'purchase_form_key' not in st.session_state: # Xarid formasini dinamik boshqarish
    st.session_state.purchase_form_key = 0
# `products_in_cart` ni endi saqlash shart emas


# --- Ma'lum yuzlarni yuklash ---
@st.cache_data(ttl=300) # 5 daqiqada bir marta keshni yangilash
def load_known_faces():
    known_face_encodings = []
    known_face_ids = []
    try:
        customers_with_embeddings = db.get_all_customer_embeddings()
        for customer in customers_with_embeddings:
            known_face_encodings.append(customer['embedding'])
            known_face_ids.append(customer['id'])
    except Exception as e:
        st.error(f"Yuklangan yuz embeddinglarini olishda xatolik: {e}")
    return known_face_encodings, known_face_ids

known_face_encodings, known_face_ids = load_known_faces()
st.sidebar.info(f"Tanish uchun {len(known_face_ids)} ta yuz yuklandi.")

# --- Kunlik hisobotlarni olish funksiyasi ---
@st.cache_data(ttl=60) # Har 1 daqiqada yangilash
def get_daily_report(date_str):
    conn = None
    cur = None
    total_daily_sales = 0.0
    daily_purchases = []
    try:
        conn = db.connect_db()
        cur = conn.cursor()
        
        # Kunlik jami savdoni olish
        cur.execute("""
            SELECT SUM(total_amount) AS total_sales
            FROM purchases
            WHERE SUBSTR(timestamp, 1, 10) = %s;
        """, (date_str,))
        total_sales_row = cur.fetchone()
        if total_sales_row and total_sales_row['total_sales'] is not None:
            total_daily_sales = float(total_sales_row['total_sales'])

        # So'nggi 5 ta xaridni olish
        cur.execute("""
            SELECT p.timestamp, p.total_amount, p.product_list_json, c.name AS customer_name
            FROM purchases p
            JOIN customers c ON p.customer_id = c.id
            WHERE SUBSTR(p.timestamp, 1, 10) = %s
            ORDER BY p.timestamp DESC LIMIT 5;
        """, (date_str,))
        daily_purchases = cur.fetchall()

    except Exception as e:
        st.error(f"❌ Kunlik hisobotlarni olishda xatolik: {e}")
    finally:
        if cur: cur.close()
        if conn: conn.close()
    return total_daily_sales, daily_purchases


with st.sidebar:
    st.header("Admin Sozlamalari")
    st.write("Bu yerda umumiy ma'lumotlar yoki tezkor boshqaruv joylashishi mumkin.")

    st.subheader("📊 Umumiy ma'lumotlar")
    try:
        conn = db.connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(id) AS count FROM customers")
        total_customers = cursor.fetchone()['count']
        cursor.execute("SELECT COALESCE(SUM(total_visits), 0) AS sum FROM customers")
        total_visits = cursor.fetchone()['sum']
        cursor.execute("SELECT COALESCE(SUM(total_amount_spent), 0.0) AS sum FROM customers")
        total_spent = cursor.fetchone()['sum']
        conn.close()

        st.metric("Jami mijozlar", total_customers)
        st.metric("Jami tashriflar", total_visits)
        st.metric("Jami sarflangan", f"${total_spent:.2f}")

        st.subheader("So'nggi tashriflar")
        latest_customers_sidebar = db.get_latest_customers(limit=3)
        if latest_customers_sidebar:
            for customer in latest_customers_sidebar:
                st.markdown(f"**{customer['name']}** (ID: {customer['id']}) - Sarf: ${float(customer['total_amount_spent']):.2f}")
        else:
            st.info("Hozircha mijozlar yo'q.")
    except Exception as e:
        st.error(f"❌ Umumiy ma'lumotlarni yuklashda xatolik: {e}")

    st.markdown("---")
    st.subheader("🗓️ Kunlik Hisobotlar")
    today_date = datetime.now().strftime("%Y-%m-%d")
    total_sales_today, latest_daily_purchases = get_daily_report(today_date)
    st.metric(f"Bugungi umumiy savdo ({today_date})", f"${total_sales_today:.2f}")

    if latest_daily_purchases:
        st.write("So'nggi xaridlar:")
        for purchase in latest_daily_purchases:
            # Mahsulot listini ko'rsatish shart emas, chunki endi faqat total_amount kiritiladi.
            # Agar POS dan keladigan mahsulot nomi ham ko'rsatilishi kerak bo'lsa, bu yerda o'zgartirish kerak.
            products_info = json.loads(purchase['product_list_json']) if purchase['product_list_json'] else []
            # Agar mahsulotlar ro'yxati POS dan kelmasa, "Noma'lum mahsulotlar" yoki shunga o'xshash yoziladi
            # Misol uchun, agar product_list_json har doim bo'sh list bo'lsa:
            # product_names = "N/A"
            # Agar siz POS dan kelgan mahsulot ro'yxatini ham saqlamoqchi bo'lsangiz, uni shu yerda pars qilib ko'rsating.
            product_names = ", ".join([p.get('name', 'N/A') for p in products_info]) if products_info else "N/A"

            st.markdown(f"- **{purchase['customer_name']}**: ${float(purchase['total_amount']):.2f} ({product_names})")
    else:
        st.info("Bugun hech qanday xarid qayd etilmagan.")

    st.markdown("---")
    st.subheader("Ilova boshqaruvi")

    def stop_camera_on_shutdown():
        if 'camera_running' in st.session_state and st.session_state.camera_running:
            st.session_state.camera_running = False
            st.warning("Kamera to'xtatildi. Ilova yopilmoqda...")

    if st.button("🔴 Seansni Yakunlash", key="end_session_btn", help="Ilovani to'xtatish va terminalni yopish"):
        stop_camera_on_shutdown()
        st.info("Ilova to'xtatildi. Terminalda jarayon yakunlanmoqda...")
        sys.exit(0)


col_live_stream, col_customer_details = st.columns([2, 1])

with col_live_stream:
    st.header("📹 Jonli kuzatuv (Yuzni tanish)")

    col_cam_btns1, col_cam_btns2 = st.columns(2)
    with col_cam_btns1:
        if st.button("Kamerani ishga tushirish", key="start_camera_btn", disabled=st.session_state.camera_running):
            st.session_state.camera_running = True
    with col_cam_btns2:
        if st.button("Kamerani to'xtatish", key="stop_camera_btn", disabled=not st.session_state.camera_running):
            st.session_state.camera_running = False

    st.warning("Eslatma: Bu 'jonli efir' uchun oddiy simulyatsiya. Haqiqiy va uzluksiz video oqimi uchun `streamlit-webrtc` kabi kutubxonalar tavsiya etiladi.")
    st.subheader("Kamera tasviri:")
    frame_placeholder = st.empty()

    camera = None

    if st.session_state.camera_running:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            st.error("Kamera topilmadi yoki ochilmadi. Iltimos, boshqa dasturlar kameradan foydalanmayotganligiga ishonch hosil qiling.")
            st.session_state.camera_running = False
        else:
            while st.session_state.camera_running:
                ret, frame = camera.read()
                if not ret:
                    st.error("Kameradan kadr olishda xatolik.")
                    break

                faces = face_analyser.get(frame)

                detected_id = None
                detected_name = "Noma'lum"
                current_face_embedding = None

                if faces:
                    face = faces[0]
                    current_face_embedding_raw = face.embedding
                    current_face_embedding = db.l2_normalize(current_face_embedding_raw)
                    
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

                    img_h, img_w, _ = frame.shape
                    face_img_cropped = frame[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
                    
                    if current_face_embedding is None or not face_img_cropped.size > 0:
                        detected_name = "Xato!"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, detected_name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 1)
                        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                        continue

                    match_threshold = 0.65 
                    best_match_id = None
                    min_distance = float('inf')

                    for known_id, known_embedding in zip(known_face_ids, known_face_encodings):
                        distance = np.linalg.norm(current_face_embedding - known_embedding)

                        if distance < min_distance:
                            min_distance = distance
                            best_match_id = known_id
                    
                    if min_distance < match_threshold:
                        detected_id = best_match_id
                        conn_temp = None
                        try:
                            conn_temp = db.connect_db()
                            cursor_temp = conn_temp.cursor()
                            cursor_temp.execute("SELECT name FROM customers WHERE id = %s", (detected_id,))
                            name_from_db_row = cursor_temp.fetchone()
                            if name_from_db_row:
                                detected_name = name_from_db_row['name']
                        except Exception as e:
                            print(f"Xatolik: Mijoz nomini yuklashda (kamera loopida): {e}")
                        finally:
                            if conn_temp: conn_temp.close()
                    else:
                        detected_name = "Noma'lum"
                        current_time = datetime.now()
                        
                        if (current_time - st.session_state.last_unknown_detection_time).total_seconds() > st.session_state.unknown_detection_cooldown:
                            st.session_state.add_new_customer_prompt = True
                            st.session_state.new_customer_embedding_temp = current_face_embedding
                            
                            temp_image_name = f"unknown_face_{current_time.strftime('%Y%m%d%H%M%S')}.jpg"
                            temp_image_path = os.path.join(CUSTOMER_PROFILE_DIR, temp_image_name)
                            cv2.imwrite(temp_image_path, cv2.cvtColor(face_img_cropped, cv2.COLOR_RGB2BGR))
                            st.session_state.new_customer_image_temp = temp_image_path

                            st.session_state.last_unknown_detection_time = current_time
                            st.rerun()

                    color = (0, 255, 0) if detected_id else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, detected_name, (x1 + 6, y2 - 6), font, 0.7, (255, 255, 255), 1)

                frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                if st.session_state.detected_customer_id != detected_id:
                    st.session_state.detected_customer_id = detected_id
                    st.session_state.current_customer_id = detected_id
                    st.session_state.current_customer_name = detected_name
                    # st.session_state.products_in_cart = [] # Endi kerak emas
                    st.session_state.purchase_form_key += 1 # Xarid formasini yangilash uchun key ni oshiramiz
                    
                    if detected_id is not None:
                        st.session_state.add_new_customer_prompt = False
                        st.session_state.new_customer_embedding_temp = None
                        if st.session_state.new_customer_image_temp and os.path.exists(st.session_state.new_customer_image_temp):
                            os.remove(st.session_state.new_customer_image_temp)
                            st.session_state.new_customer_image_temp = None
                        
                        try:
                            db.record_visit(detected_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                            st.toast(f"{detected_name} ({detected_id}) uchun tashrif qayd etildi!")
                        except Exception as e:
                            st.error(f"Tashrifni qayd etishda xatolik: {e}")
                    
                    st.rerun()

    if camera and not st.session_state.camera_running:
        camera.release()
        cv2.destroyAllWindows()
        st.info("Kamera o'chirildi.")


with col_customer_details:
    st.header("Mijoz ma'lumotlari")

    # --- Yangi mijoz qo'shish formasi ---
    if st.session_state.add_new_customer_prompt:
        st.subheader("🆕 Yangi Noma'lum Mijozni Qo'shish")
        st.warning("Kamerada noma'lum yuz aniqlandi. Uni tizimga qo'shishni xohlaysizmi?")
        
        if st.session_state.new_customer_image_temp and os.path.exists(st.session_state.new_customer_image_temp):
            st.image(display_image_from_path(st.session_state.new_customer_image_temp), caption="Aniqlangan yuz", use_container_width=True)
            
        with st.form("quick_add_new_customer_form", clear_on_submit=True):
            new_customer_name_quick = st.text_input("Ism:", key="new_name_quick_add")
            new_customer_phone_quick = st.text_input("Telefon raqami:", key="new_phone_quick_add")
            new_customer_email_quick = st.text_input("Email (ixtiyoriy):", key="new_email_quick_add")

            col_add_confirm, col_add_cancel = st.columns(2)
            with col_add_confirm:
                submit_quick_add = st.form_submit_button("✅ Mijozni Qo'shish") 
            with col_add_cancel:
                cancel_quick_add = st.form_submit_button("❌ Bekor Qilish") 

            if submit_quick_add:
                if not new_customer_name_quick:
                    st.error("Ism kiritish majburiy!")
                else:
                    try:
                        new_customer_id = db.add_customer(
                            name=new_customer_name_quick,
                            phone_number=new_customer_phone_quick,
                            email=new_customer_email_quick,
                            profile_image_path=st.session_state.new_customer_image_temp,
                            embedding=st.session_state.new_customer_embedding_temp
                        )
                        
                        if st.session_state.new_customer_image_temp and new_customer_id:
                            file_extension = st.session_state.new_customer_image_temp.split('.')[-1]
                            final_image_name = f"customer_{new_customer_id}.{file_extension}"
                            final_profile_image_path = os.path.join(CUSTOMER_PROFILE_DIR, final_image_name)
                            
                            os.rename(st.session_state.new_customer_image_temp, final_profile_image_path)
                            
                            db.update_customer_data(
                                customer_id=new_customer_id,
                                new_name=new_customer_name_quick,
                                new_phone=new_customer_phone_quick,
                                new_email=new_customer_email_quick,
                                new_profile_image_path=final_profile_image_path,
                                updated_embedding=st.session_state.new_customer_embedding_temp
                            )

                        st.success(f"Yangi mijoz '{new_customer_name_quick}' muvaffaqiyatli qo'shildi! ID: {new_customer_id}")
                        st.session_state.current_customer_id = new_customer_id
                        st.session_state.add_new_customer_prompt = False
                        st.session_state.new_customer_embedding_temp = None
                        st.session_state.new_customer_image_temp = None
                        st.session_state.last_unknown_detection_time = datetime.now()
                        # st.session_state.products_in_cart = [] # Endi kerak emas
                        st.session_state.purchase_form_key += 1 # Xarid formasini yangilash
                        
                        load_known_faces.clear() 
                        get_daily_report.clear() # Kunlik hisobot keshini tozalash
                        st.rerun() 
                        
                    except Exception as e:
                        st.error(f"❌ Yangi mijoz qo'shishda xatolik: {e}")
                        if st.session_state.new_customer_image_temp and os.path.exists(st.session_state.new_customer_image_temp):
                            os.remove(st.session_state.new_customer_image_temp)

            elif cancel_quick_add:
                st.session_state.add_new_customer_prompt = False
                st.session_state.new_customer_embedding_temp = None
                if st.session_state.new_customer_image_temp and os.path.exists(st.session_state.new_customer_image_temp):
                    os.remove(st.session_state.new_customer_image_temp)
                    st.session_state.new_customer_image_temp = None
                st.session_state.last_unknown_detection_time = datetime.now()
                st.rerun()
    else: # Agar avtomatik qo'shish formasi ko'rsatilmasa, odatiy mijoz boshqaruvi
        if st.session_state.detected_customer_id:
            st.info(f"Kamerada aniqlangan mijoz: ID **{st.session_state.detected_customer_id}** ({st.session_state.current_customer_name})")
            customer_id_search_default = st.session_state.detected_customer_id
        else:
            customer_id_search_default = st.session_state.current_customer_id if st.session_state.current_customer_id is not None else 1

        st.subheader("Mijozni ID bo'yicha qidirish / Tahrirlash")
        customer_id_input = st.number_input("Mijoz ID sini kiriting:", min_value=1, value=customer_id_search_default, step=1, key="manual_customer_id_input")

        if st.button("Mijozni yuklash", key="load_customer_btn"):
            st.session_state.current_customer_id = customer_id_input
            st.session_state.detected_customer_id = None
            # st.session_state.products_in_cart = [] # Endi kerak emas
            st.session_state.purchase_form_key += 1 # Xarid formasini yangilash
            st.rerun()

        customer_profile = None
        visits_data = None
        purchases_data = None

        if st.session_state.current_customer_id:
            customer_profile, visits_data, purchases_data = db.get_customer_full_profile(st.session_state.current_customer_id)

        if customer_profile:
            st.write("### Mijoz profili")
            st.write(f"**ID:** {customer_profile['id']}")
            st.write(f"**Ism:** {customer_profile['name']}")
            st.write(f"**Telefon:** {customer_profile.get('phone_number', 'Kiritilmagan')}")
            st.write(f"**Email:** {customer_profile.get('email', 'Kiritilmagan')}")
            st.write(f"**Birinchi tashrif:** {customer_profile['first_visit_time']}")
            st.write(f"**Oxirgi tashrif:** {customer_profile['last_visit_time']}")
            st.write(f"**Jami tashriflar:** {customer_profile['total_visits']}")
            st.write(f"**Jami sarflangan:** ${float(customer_profile['total_amount_spent']):.2f}")
            st.write(f"**Cashback ballari:** {float(customer_profile['cashback_points']):.2f}")

            if customer_profile['profile_image_path']:
                st.image(display_image_from_path(customer_profile['profile_image_path']), caption="Profil rasmi", use_container_width=True)

            st.subheader("Mijoz ma'lumotlarini yangilash")
            with st.form("update_customer_form"):
                new_name = st.text_input("Ism:", value=customer_profile['name'])
                new_phone = st.text_input("Telefon raqami:", value=customer_profile.get('phone_number', ''))
                new_email = st.text_input("Email:", value=customer_profile.get('email', ''))

                uploaded_file = st.file_uploader("Yangi profil rasmini yuklash:", type=["jpg", "png", "jpeg"])
                
                submit_update = st.form_submit_button("Ma'lumotlarni yangilash")

                if submit_update:
                    updated_image_path = customer_profile.get('profile_image_path')
                    updated_embedding = None

                    if uploaded_file is not None:
                        if updated_image_path and os.path.exists(updated_image_path):
                            try:
                                os.remove(updated_image_path)
                            except Exception as e:
                                st.warning(f"Eski rasmni o'chirishda xatolik: {e}")

                        file_extension = uploaded_file.name.split('.')[-1]
                        new_image_name = f"customer_{customer_profile['id']}_{datetime.now().strftime('%Y%m%d%H%M%S')}.{file_extension}"
                        updated_image_path = os.path.join(CUSTOMER_PROFILE_DIR, new_image_name)
                        with open(updated_image_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        st.success("Yangi rasm muvaffaqiyatli yuklandi.")

                        try:
                            image = cv2.imread(updated_image_path)
                            if image is None: raise ValueError("Rasm yuklanmadi.")
                            faces_in_uploaded = face_analyser.get(image)
                            if faces_in_uploaded:
                                updated_embedding = db.l2_normalize(faces_in_uploaded[0].embedding) 
                                st.write(f"DEBUG: Yangilangan embedding hisoblandi. O'lchami: {len(updated_embedding)}")
                            else:
                                st.warning("Yuklangan yangi rasmdan yuz topilmadi. Mijoz yuzi aniqlanmaydi.")
                        except Exception as e:
                            st.error(f"Yangilangan rasm embeddingini hisoblashda xatolik: {e}")
                            updated_embedding = None
                    else:
                        if 'face_embedding' in customer_profile and customer_profile['face_embedding'] is not None:
                            try:
                                loaded_embedding_raw = np.array(json.loads(customer_profile['face_embedding']))
                                updated_embedding = db.l2_normalize(loaded_embedding_raw)
                            except json.JSONDecodeError as e:
                                st.error(f"Eski embeddingni yuklashda xatolik: {e}")
                                updated_embedding = None

                    try:
                        db.update_customer_data(customer_profile['id'], new_name, new_phone, new_email, updated_image_path, updated_embedding)
                        st.success(f"Mijoz ID {customer_profile['id']} ma'lumotlari yangilandi!")
                        st.session_state.current_customer_name = new_name
                        load_known_faces.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Ma'lumotlarni yangilashda xatolik: {e}")

            st.write("#### Oxirgi tashriflar")
            if visits_data:
                for visit in visits_data:
                    st.write(f"- {visit['timestamp']} | Kayfiyat: {visit.get('emotion', 'N/A')} | Jins: {visit.get('gender', 'N/A')} | Yosh: {visit.get('age', 'N/A')}")
            else:
                st.info("Tashriflar topilmadi.")

            st.write("#### Oxirgi xaridlar")
            if purchases_data:
                for purchase in purchases_data:
                    products = json.loads(purchase['product_list_json']) if purchase['product_list_json'] else []
                    # Agar mahsulot listi bo'lmasa, shunchaki "N/A" yozamiz
                    product_names_str = ', '.join([p.get('name', 'N/A') for p in products]) if products else "N/A"
                    st.write(f"- {purchase['timestamp']} | Summa: ${float(purchase['total_amount']):.2f} | Mahsulotlar: {product_names_str}")
            else:
                st.info("Xaridlar topilmadi.")

            # --- Yangi Xarid Qayd Etish Formasi (Soddalashtirilgan) ---
            st.subheader(f"🛒 {customer_profile['name']} uchun xaridni qayd etish")

            with st.form(key=f"record_purchase_form_{st.session_state.purchase_form_key}"):
                
                # Faqat xarid miqdorini kiritish maydoni
                total_amount_input = st.number_input("Xaridning umumiy miqdori ($):", min_value=0.01, format="%.2f", key="total_amount_input_key")

                payment_method = st.selectbox("To'lov usuli:", ["Naqd", "Karta", "Pul o'tkazmasi"], key="payment_method_input")
                
                record_purchase_submit = st.form_submit_button("✅ Xaridni qayd etish")

                if record_purchase_submit:
                    if total_amount_input <= 0:
                        st.error("Iltimos, umumiy miqdorni to'g'ri kiriting.")
                    else:
                        try:
                            current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            
                            # Mahsulotlar ro'yxatini bo'sh yoki shunchaki umumiy miqdor haqidagi ma'lumot sifatida saqlash
                            # Agar POS sistemadan faqat umumiy narx kelishi kutilsa, bu yerda bo'sh list saqlanishi mumkin.
                            # Yoki keyinchalik bu yerga POS dan keladigan mahsulot listini qo'yishingiz mumkin.
                            # Hozircha bo'sh list saqlaymiz:
                            product_list_to_save = [] 
                            
                            db.record_purchase(
                                customer_profile['id'],
                                current_time_str,
                                total_amount_input, # Kiritilgan umumiy miqdor
                                product_list_to_save, # Bo'sh mahsulot ro'yxati
                                payment_method
                            )
                            st.success(f"Xarid muvaffaqiyatli qayd etildi! Jami: ${total_amount_input:.2f}")
                            st.session_state.purchase_form_key += 1 # Formani yangilash
                            get_daily_report.clear() # Kunlik hisobot keshini tozalash
                            st.rerun() # UI ni yangilash
                        except Exception as e:
                            st.error(f"❌ Xaridni qayd etishda xatolik: {e}")
            # --- Xarid Qayd Etish Formasi Tugadi ---

        else:
            st.warning("Mijoz topilmadi yoki hali tanlanmadi.")

        st.subheader("Yangi mijoz qo'shish (Qo'lda)")
        with st.form("add_new_customer_form", clear_on_submit=True):
            new_customer_name = st.text_input("Ism:", key="manual_add_name")
            new_customer_phone = st.text_input("Telefon raqami:", key="manual_add_phone")
            new_customer_email = st.text_input("Email:", key="manual_add_email")
            new_customer_photo = st.file_uploader("Profil rasmini yuklash:", type=["jpg", "png", "jpeg"], key="manual_add_photo")

            submit_new = st.form_submit_button("Yangi mijozni qo'shish")

            if submit_new:
                if not new_customer_name:
                    st.error("Ism kiritish majburiy!")
                else:
                    profile_image_path = None
                    temp_profile_image_path = None
                    file_extension = None
                    new_face_embedding = None

                    if new_customer_photo is not None:
                        if not os.path.exists(CUSTOMER_PROFILE_DIR):
                            os.makedirs(CUSTOMER_PROFILE_DIR)
                        
                        file_extension = new_customer_photo.name.split('.')[-1]
                        temp_image_name = f"temp_customer_{datetime.now().strftime('%Y%m%d%H%M%S')}.{file_extension}"
                        temp_profile_image_path = os.path.join(CUSTOMER_PROFILE_DIR, temp_image_name)
                        with open(temp_profile_image_path, "wb") as f:
                            f.write(new_customer_photo.getbuffer())
                        profile_image_path = temp_profile_image_path

                        try:
                            image = cv2.imread(temp_profile_image_path)
                            if image is None: raise ValueError("Rasm yuklanmadi.")
                            faces_in_new_customer = face_analyser.get(image)
                            if faces_in_new_customer:
                                new_face_embedding = db.l2_normalize(faces_in_new_customer[0].embedding)
                                st.write(f"DEBUG: Yangi mijoz uchun embedding hisoblandi. O'lchami: {len(new_face_embedding)}")
                            else:
                                st.warning("Yuklangan rasmdan yuz topilmadi. Mijoz yuzi aniqlanmaydi.")
                        except Exception as e:
                            st.error(f"Rasm embeddingini hisoblashda xatolik: {e}")
                            new_face_embedding = None

                    try:
                        new_customer_id = db.add_customer(
                            name=new_customer_name,
                            phone_number=new_customer_phone,
                            email=new_customer_email,
                            profile_image_path=profile_image_path,
                            embedding=new_face_embedding
                        )
                        
                        if temp_profile_image_path and new_customer_id:
                            final_image_name = f"customer_{new_customer_id}.{file_extension}"
                            final_profile_image_path = os.path.join(CUSTOMER_PROFILE_DIR, final_image_name)
                            os.rename(temp_profile_image_path, final_profile_image_path)
                            db.update_customer_data(
                                customer_id=new_customer_id,
                                new_name=new_customer_name,
                                new_phone=new_customer_phone,
                                new_email=new_customer_email,
                                new_profile_image_path=final_profile_image_path,
                                updated_embedding=new_face_embedding
                            )

                        st.success(f"Yangi mijoz '{new_customer_name}' muvaffaqiyatli qo'shildi! ID: {new_customer_id}")
                        st.session_state.current_customer_id = new_customer_id
                        load_known_faces.clear()
                        get_daily_report.clear() # Kunlik hisobot keshini tozalash
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Yangi mijoz qo'shishda xatolik: {e}")
                        if temp_profile_image_path and os.path.exists(temp_profile_image_path):
                            os.remove(temp_profile_image_path)