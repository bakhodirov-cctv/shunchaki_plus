import psycopg2
import psycopg2.extras # lug'at (dictionary) shaklida natija olish uchun
import json
import numpy as np
from datetime import datetime

# --- YANGI FUNKSIYA: L2-normalizatsiya (Chunki DB'dan olingan embeddinglar ham normallashtirilishi kerak) ---
def l2_normalize(x):
    """ NumPy massivini L2-normalizatsiya qiladi. """
    norm = np.linalg.norm(x)
    return x / norm if norm != 0 else x
# --- YANGI FUNKSIYA TUGADI ---

# Ma'lumotlar bazasi ulanish parametrlari
db_params = {
    "host": "localhost",
    "database": "python", # <<< SIZNING MA'LUMOTLAR BAZANGIZ NOMINI KIRITING!
    "user": "postgres",
    "password": "Asadbek2005@", # <<< SIZNING PAROLINGIZNI KIRITING!
    "port": "5432"
}

# --- Ma'lumotlar bazasi ulanish funksiyasi ---
def connect_db():
    """PostgreSQL ga ulanishni yaratadi va qaytaradi."""
    conn = None
    try:
        # RealDictCursor - natijalarni lug'at (dictionary) sifatida olish uchun
        conn = psycopg2.connect(**db_params, cursor_factory=psycopg2.extras.RealDictCursor)
        # SET search_path ni aniq belgilash (PostgreSQL session uchun)
        cursor = conn.cursor()
        cursor.execute("SET search_path TO public;")
        cursor.close()
        return conn
    except psycopg2.Error as e:
        print(f"❌ DB ulanishda xatolik yuz berdi: {e}")
        raise

# --- Ma'lumotlar bazasi jadvallarini yaratish funksiyasi ---
def create_tables():
    """
    Ma'lumotlar bazasida jadvallarni yaratadi (agar mavjud bo'lmasa)
    va mavjud 'customers' jadvaliga yangi ustunlarni qo'shadi (faqat bir marta).
    """
    conn = None
    cur = None
    try:
        conn = connect_db()
        cur = conn.cursor()

        # 1. Customers jadvalini yaratish
        cur.execute('''
            CREATE TABLE IF NOT EXISTS customers (
                id SERIAL PRIMARY KEY,
                name TEXT DEFAULT 'Noma''lum',
                first_visit_time TEXT NOT NULL,
                last_visit_time TEXT NOT NULL,
                total_visits INTEGER DEFAULT 1,
                face_embedding TEXT, -- NOT NULL cheklovi o'chirilgan, chunki har doim ham bo'lmasligi mumkin
                dominant_gender TEXT,
                average_age INTEGER,
                total_amount_spent NUMERIC(10, 2) DEFAULT 0.0,
                cashback_points NUMERIC(10, 2) DEFAULT 0.0,
                profile_image_path TEXT,
                phone_number TEXT,  -- Yangi ustun
                email TEXT          -- Yangi ustun
            )
        ''')
        conn.commit()
        print("✅ Customers jadvali tayyor.")

        # 2. Visits jadvalini yaratish
        cur.execute('''
            CREATE TABLE IF NOT EXISTS visits (
                id SERIAL PRIMARY KEY,
                customer_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                emotion TEXT,
                gender TEXT,
                age INTEGER,
                FOREIGN KEY (customer_id) REFERENCES customers (id) ON DELETE CASCADE
            )
        ''')
        conn.commit()
        print("✅ Visits jadvali tayyor.")

        # 3. Purchases jadvalini yaratish
        cur.execute('''
            CREATE TABLE IF NOT EXISTS purchases (
                id SERIAL PRIMARY KEY,
                customer_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                receipt_number TEXT UNIQUE,
                total_amount NUMERIC(10, 2) NOT NULL,
                product_list_json TEXT,
                payment_method TEXT,
                FOREIGN KEY (customer_id) REFERENCES customers (id) ON DELETE CASCADE
            )
        ''')
        conn.commit()
        print("✅ Purchases jadvali tayyor.")

        # Izoh: Oldingi ALTER TABLE buyruqlari olib tashlangan,
        # chunki ustunlar endi CREATE TABLE ichida e'lon qilingan.
        # Agar sizda eski jadvalda (CREATE TABLE ga qaramay) ustunlar yo'q bo'lsa
        # va siz ma'lumotlarni yo'qotmasdan yangilamoqchi bo'lsangiz,
        # bu COMMENTED OUT qismlarni bir marta ishlating va keyin yana COMMENTED OUT qilib qo'ying.

        # try:
        #     cur.execute("ALTER TABLE customers ADD COLUMN phone_number TEXT;")
        #     conn.commit()
        #     print("✅ 'phone_number' ustuni 'customers' jadvaliga qo'shildi (ALTERED).")
        # except psycopg2.ProgrammingError as e:
        #     if "column \"phone_number\" already exists" in str(e):
        #         conn.rollback()
        #         print("'phone_number' ustuni allaqachon mavjud.")
        #     else:
        #         conn.rollback()
        #         raise e

        # try:
        #     cur.execute("ALTER TABLE customers ADD COLUMN email TEXT;")
        #     conn.commit()
        #     print("✅ 'email' ustuni 'customers' jadvaliga qo'shildi (ALTERED).")
        # except psycopg2.ProgrammingError as e:
        #     if "column \"email\" already exists" in str(e):
        #         conn.rollback()
        #         print("'email' ustuni allaqachon mavjud.")
        #     else:
        #         conn.rollback()
        #         raise e

        # try:
        #     cur.execute("ALTER TABLE customers ALTER COLUMN face_embedding DROP NOT NULL;")
        #     conn.commit()
        #     print("✅ 'face_embedding' ustuni NOT NULL cheklovi olib tashlandi (ALTERED).")
        # except psycopg2.ProgrammingError as e:
        #     if "column \"face_embedding\" is not null" in str(e).lower() or "not null" not in str(e).lower():
        #         conn.rollback()
        #         print("Eslatma: 'face_embedding' ustuni allaqachon NULL bo'lishi mumkin.")
        #     else:
        #         conn.rollback()
        #         raise e


    except Exception as e:
        print(f"❌ Ma'lumotlar bazasi jadvallarini yaratishda/yangilashda xatolik: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if cur: cur.close()
        if conn: conn.close()

# --- Mijozni qo'shish funksiyasi ---
def add_customer(name, phone_number=None, email=None, profile_image_path=None, embedding=None):
    conn = None
    cur = None
    try:
        conn = connect_db()
        cur = conn.cursor()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        embedding_str = json.dumps(embedding.tolist()) if embedding is not None else None

        cur.execute('''
            INSERT INTO customers (name, phone_number, email, profile_image_path,
                                   first_visit_time, last_visit_time, total_visits,
                                   face_embedding, dominant_gender, average_age)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
        ''', (name, phone_number, email, profile_image_path,
              current_time, current_time, 1,
              embedding_str, None, None)) # dominant_gender va average_age boshida None bo'lishi mumkin
        customer_id = cur.fetchone()['id']
        conn.commit()
        print(f"✅ Yangi mijoz qo'shildi: ID {customer_id}, Ism: {name}")
        return customer_id
    except psycopg2.Error as e:
        print(f"❌ Yangi mijoz qo'shishda xatolik: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if cur: cur.close()
        if conn: conn.close()

# --- Mijoz profilini yangilash funksiyasi ---
def update_customer_data(customer_id, new_name, new_phone, new_email, new_profile_image_path, updated_embedding=None):
    conn = None
    cur = None
    try:
        conn = connect_db()
        cur = conn.cursor()

        embedding_str = json.dumps(updated_embedding.tolist()) if updated_embedding is not None else None

        cur.execute('''
            UPDATE customers
            SET name = %s,
                phone_number = %s,
                email = %s,
                profile_image_path = %s,
                face_embedding = %s -- face_embedding ustunini yangilash
            WHERE id = %s
        ''', (new_name, new_phone, new_email, new_profile_image_path, embedding_str, customer_id))
        conn.commit()
        print(f"✅ Mijoz ID {customer_id} ma'lumotlari yangilandi.")
    except psycopg2.Error as e:
        print(f"❌ Mijoz ma'lumotlarini yangilashda xatolik: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if cur: cur.close()
        if conn: conn.close()

# --- Barcha mijozlarning embeddinglarini olish funksiyasi ---
def get_all_customer_embeddings():
    conn = None
    cur = None
    try:
        conn = connect_db()
        cur = conn.cursor()
        cur.execute("SELECT id, face_embedding FROM customers WHERE face_embedding IS NOT NULL") # Faqat embedding bor mijozlarni olish
        customers = []
        for row in cur.fetchall():
            loaded_embedding = np.array(json.loads(row['face_embedding']))
            normalized_embedding = l2_normalize(loaded_embedding) # <-- Bazadan o'qilgan embeddingni normallashtiramiz
            customers.append({
                "id": row['id'],
                "embedding": normalized_embedding
            })
        return customers
    except psycopg2.Error as e:
        print(f"❌ Mijoz embeddinglarini olishda xatolik: {e}")
        raise
    finally:
        if cur: cur.close()
        if conn: conn.close()

# --- Tashrifni qayd etish funksiyasi ---
def record_visit(customer_id, timestamp_str, emotion=None, gender=None, age=None):
    conn = None
    cur = None
    try:
        conn = connect_db()
        cur = conn.cursor()
        cur.execute("INSERT INTO visits (customer_id, timestamp, emotion, gender, age) VALUES (%s, %s, %s, %s, %s);",
                       (customer_id, timestamp_str, emotion, gender, age))

        cur.execute("""
            UPDATE customers
            SET total_visits = total_visits + 1,
                last_visit_time = %s
            WHERE id = %s;
        """, (timestamp_str, customer_id))
        conn.commit()
        print(f"✅ Mijoz ID {customer_id} uchun tashrif qayd etildi.")
    except psycopg2.Error as e:
        print(f"❌ Tashrifni qayd etishda xatolik: {e}")
        if conn: conn.rollback()
        raise
    finally:
        if cur: cur.close()
        if conn: conn.close()

# --- Xaridni qayd etish funksiyasi ---
def record_purchase(customer_id, timestamp, total_amount, product_list, payment_method="Naqd"):
    conn = None
    cur = None
    try:
        conn = connect_db()
        cur = conn.cursor()
        product_list_json = json.dumps(product_list)
        receipt_num = f"REC_{timestamp.replace(' ', '_').replace(':', '-')}_{customer_id}"

        cur.execute('''
            INSERT INTO purchases (customer_id, timestamp, receipt_number, total_amount, product_list_json, payment_method)
            VALUES (%s, %s, %s, %s, %s, %s)
        ''', (customer_id, timestamp, receipt_num, total_amount, product_list_json, payment_method))

        cur.execute('UPDATE customers SET total_amount_spent = total_amount_spent + %s, cashback_points = cashback_points + %s WHERE id = %s',
                       (total_amount, total_amount * 0.01, customer_id))

        conn.commit()
        print(f"💰 Mijoz ID {customer_id} uchun {total_amount} summa xarid qayd etildi. Chek: {receipt_num}")
    except psycopg2.Error as e:
        print(f"❌ Xaridni qayd etishda xatolik: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if cur: cur.close()
        if conn: conn.close()

# --- Mijozning to'liq profilini olish funksiyasi ---
def get_customer_full_profile(customer_id):
    conn = None
    cur = None
    customer_profile = None
    visits_data = []
    purchases_data = []
    try:
        conn = connect_db()
        cur = conn.cursor() # RealDictCursor dan foydalanilgan

        print(f"DEBUG DB: Mijoz ID {customer_id} uchun profilni yuklash...")
        cur.execute("SELECT * FROM customers WHERE id = %s", (customer_id,))
        customer_profile = cur.fetchone()
        print(f"DEBUG DB: Mijoz profili yuklandi: {customer_profile is not None}")

        if customer_profile:
            print("DEBUG DB: Tashriflar ma'lumotini yuklash...")
            try:
                cur.execute("SELECT * FROM visits WHERE customer_id = %s ORDER BY timestamp DESC LIMIT 3", (customer_id,))
                visits_data = cur.fetchall() # RealDictCursor tufayli lug'atlar ro'yxati
                print(f"DEBUG DB: Tashriflar ma'lumoti yuklandi. Soni: {len(visits_data)}")
            except Exception as e:
                print(f"❌ DEBUG DB: Tashriflar ma'lumotini olishda xatolik: {e}")
                visits_data = []

            print("DEBUG DB: Xaridlar ma'lumotini yuklash...")
            try:
                cur.execute("SELECT * FROM purchases WHERE customer_id = %s ORDER BY timestamp DESC LIMIT 3", (customer_id,))
                purchases_data = cur.fetchall() # RealDictCursor tufayli lug'atlar ro'yxati
                print(f"DEBUG DB: Xaridlar ma'lumoti yuklandi. Soni: {len(purchases_data)}")
            except Exception as e:
                print(f"❌ DEBUG DB: Xaridlar ma'lumotini olishda xatolik: {e}")
                purchases_data = []

        print("DEBUG DB: get_customer_full_profile muvaffaqiyatli yakunlandi.")
        return customer_profile, visits_data, purchases_data
    except psycopg2.Error as e:
        print(f"❌ Mijoz profilini olishda xatolik (DB): {e}")
        return None, [], []
    except Exception as e:
        print(f"❌ Mijoz profilini olishda kutilmagan xatolik (DB): {e}")
        return None, [], []
    finally:
        if conn:
            conn.close()

def get_latest_customers(limit=5):
    """So'nggi tashrif buyurgan mijozlarni qaytaradi."""
    conn = None
    cur = None
    customers = []
    try:
        conn = connect_db()
        cur = conn.cursor() # RealDictCursor dan foydalanilgan
        cur.execute("SELECT id, name, total_visits, total_amount_spent, profile_image_path FROM customers ORDER BY last_visit_time DESC LIMIT %s;", (limit,))
        customers = cur.fetchall()
    except Exception as e:
        print(f"❌ So'nggi mijozlarni yuklashda xatolik: {e}")
        raise
    finally:
        if cur: cur.close()
        if conn: conn.close()
    return customers


# --- Boshida ma'lumotlar bazasi ulanishini tekshirish (faqat bir marta) ---
if __name__ == '__main__':
    print("Ma'lumotlar bazasi jadvallarini tekshirish/yaratish...")
    create_tables()
    print("Ma'lumotlar bazasi tayyor.")