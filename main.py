# main.py – RFMiD Retina Hastalık Tespiti Projesi

# 📦 1. Gerekli Kütüphaneler
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy, AUC
from sklearn.model_selection import train_test_split

# 📁 2. Çalışma Dizini ve Yol Ayarları
BASE_DIR = r"D:/Yapay zeka eğitme/Retinal Hastalıkların Tespiti/RFMiD"
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "Train")
TRAIN_CSV = os.path.join(DATA_DIR, "RFMiD_Training_Labels.csv")
# Test klasörü ve test etiket dosyası
TEST_DIR = os.path.join(DATA_DIR, "Test")
TEST_CSV = os.path.join(DATA_DIR, "RFMiD_Testing_Labels.csv")


# 🧪 3. Eğitim Verisini Yükle
if not os.path.exists(TRAIN_CSV):
    raise FileNotFoundError(f"Eğitim CSV dosyası bulunamadı: {TRAIN_CSV}\nLütfen dosyanın yüklenip doğru klasörde olduğundan emin olun.")

df_train = pd.read_csv(TRAIN_CSV)
print("📋 Eğitim etiketlerinden ilk 5 satır:")
print(df_train.head())

# 🔹 Sadece ilk 300 eğitim verisiyle çalış
df_train = df_train.head(200)

# 🔗 4. Görsel Yolu Ekle (PNG uzantılı görseller için)
df_train["image_path"] = df_train["ID"].apply(lambda x: os.path.join(TRAIN_DIR, f"{x}.png"))

# 🖼️ 5. Örnek Görsel Göster
sample_path = df_train["image_path"].iloc[0]
if not os.path.exists(sample_path):
    raise FileNotFoundError(f"Örnek görsel bulunamadı: {sample_path}\nLütfen 'train' klasöründeki görsellerin doğru yüklendiğini kontrol edin.")

img = Image.open(sample_path)
plt.imshow(img)
plt.title(f"Görsel: {df_train['ID'].iloc[0]}")
plt.axis("off")
plt.show()

# 🔎 6. Etiket Dağılımı Analizi
label_columns = df_train.columns[1:-1]  # Disease_Risk dahil, image_path hariç olmalı
print("Etiket sütunları:", label_columns.tolist())

df_train["label_count"] = df_train[label_columns].sum(axis=1)
print("🔢 Ortalama etiket sayısı:", df_train["label_count"].mean())

# 📊 7. Etiket Sayısı Grafiği
label_counts = df_train[label_columns].sum().sort_values(ascending=False)
plt.figure(figsize=(12, 5))
label_counts.plot(kind="bar")
plt.title("Etiket Dağılımı (Her Hastalık)")
plt.xlabel("Hastalık")
plt.ylabel("Toplam Görsel Sayısı")
plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()
plt.show()

# 📐 Parametreler
IMG_SIZE = 224

# load_data fonksiyonunu label_columns parametresi alacak şekilde güncelleyelim:
def load_data(df, label_columns=None, img_size=224):
    X = []
    y = []
    if label_columns is None:
        label_columns = [col for col in df.columns if col not in ["ID", "Disease_Risk", "image_path"]]

    print("🔄 Görseller yükleniyor...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        path = row["image_path"]
        if os.path.exists(path):
            try:
                img = Image.open(path).convert("RGB")
                img = img.resize((img_size, img_size))
                img_array = np.array(img) / 255.0  # normalize
                X.append(img_array)
                labels = np.clip(row[label_columns].astype(np.float32).values, 0, 1)
                y.append(labels)
            except Exception as e:
                print(f"⚠️ Hata oluştu: {e} - {path}")
        else:
            print(f"⚠️ Görsel bulunamadı: {path}")

    return np.array(X), np.array(y)
         
X_train, y_train = load_data(df_train)
print(f"✅ X_train şekli: {X_train.shape}")
print(f"✅ y_train şekli: {y_train.shape}")

# 📊 Eğitim/Doğrulama Ayırımı
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# 🧠 Model Oluştur
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y_train.shape[1], activation='sigmoid')
])

# ✅ Derleme
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=[BinaryAccuracy(), AUC()])

# 🚀 Modeli Eğit
history = model.fit(
    X_train_split, y_train_split,
    validation_data=(X_val_split, y_val_split),
    epochs=10,
    batch_size=32
)
print("Son epoch eğitim doğruluğu: {:.2f}%".format(history.history['binary_accuracy'][-1] * 100))
print("Son epoch doğrulama doğruluğu: {:.2f}%".format(history.history['val_binary_accuracy'][-1] * 100))


# 📈 Başarı Grafikleri
plt.plot(history.history['binary_accuracy'], label='Eğitim Doğruluk')
plt.plot(history.history['val_binary_accuracy'], label='Doğrulama Doğruluk')
plt.title('Doğruluk (Binary Accuracy)')
plt.xlabel('Epoch')
plt.ylabel('Başarı')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(history.history['loss'], label='Eğitim Kayıp')
plt.plot(history.history['val_loss'], label='Doğrulama Kayıp')
plt.title('Kayıp (Loss)')
plt.xlabel('Epoch')
plt.ylabel('Hata')
plt.legend()
plt.grid(True)
plt.show()

# 📄 Test etiketlerini yükle ve görsel yolunu tanımla
print("🔍 Test verisi hazırlanıyor...")
df_test = pd.read_csv(TEST_CSV)
# 🔹 Sadece ilk 100 test verisiyle çalış
df_test = df_test.head(100)
df_test["image_path"] = df_test["ID"].apply(lambda x: os.path.join(TEST_DIR, f"{x}.png"))

# 🛠️ Etiket sütunlarını sıfırdan kur (eğitimdeki etiketlere göre)
for col in label_columns:
    if col not in df_test.columns:
        df_test[col] = 0  # Eksik etiketleri sıfırla tamamla

# 📐 Test verisini yeniden yükle
X_test, y_test = load_data(df_test, label_columns=label_columns)

print(f"✅ X_test şekli: {X_test.shape}")
print(f"✅ y_test şekli: {y_test.shape}")
print(f"✅ X_test.shape: {X_test.shape}")
print(f"✅ y_test.shape: {y_test.shape}")


# 📊 Modeli test verisiyle değerlendir
print("📊 Test verisi değerlendiriliyor...")
results = model.evaluate(X_test, y_test, batch_size=8)


# 🧾 Metriği yazdır
for name, val in zip(model.metrics_names, results):
    if 'accuracy' in name or 'binary_accuracy' in name:
        print(f"{name}: {val * 100:.2f}%")
    else:
        print(f"{name}: {val:.4f}")

# 🔐 Eğitilen modeli .h5 dosyası olarak kaydet
model_path = os.path.join(BASE_DIR, "model", "rfmid_model.h5")
os.makedirs(os.path.dirname(model_path), exist_ok=True)  # model klasörü yoksa oluştur
model.save(model_path)

print(f"✅ Model .h5 formatında kaydedildi: {model_path}")
















