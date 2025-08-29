from flask import Flask, request, jsonify
from transformers import pipeline
import fitz 
import os
import mysql.connector
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

classifier = pipeline("zero-shot-classification", model="sberbank-ai/rugpt3large_based_on_gpt2")

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="docs_tagging"
)
cursor = db.cursor()

def extract_text(file_path):
    """Извлекаем текст из PDF с использованием fitz (PyMuPDF)"""
    text = ""
    if file_path.endswith(".pdf"):
        try:
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
        except Exception as e:
            print(f"Ошибка при извлечении текста: {e}")
    return text

def save_document(title, filename, tags):
    # Вставляем документ в таблицу documents
    cursor.execute("""
        INSERT INTO documents (title, file_path)
        VALUES (%s, %s)
    """, (title, filename))
    db.commit()
    
    # Получаем ID добавленного документа
    document_id = cursor.lastrowid

    # Вставляем теги и связываем их с документом
    for tag in tags:
        # Проверяем, существует ли тег
        cursor.execute("SELECT id FROM tags WHERE name = %s", (tag,))
        tag_id = cursor.fetchone()

        if tag_id:
            tag_id = tag_id[0]

        # Связываем тег с документом
        cursor.execute("""
            INSERT INTO document_tags (document_id, tag_id)
            VALUES (%s, %s)
        """, (document_id, tag_id))
        db.commit()

@app.route("/api/classify", methods=["POST"])
def classify_file():
    if 'file' not in request.files:
        return jsonify({"error": "Нет файла"}), 400
    
    file = request.files['file']
    filename = file.filename
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    text = extract_text(path)
    if not text.strip():
        return jsonify({"error": "Файл пуст или не удалось извлечь текст"}), 400
    cursor.execute("SELECT name FROM tags")
    tags_from_db = cursor.fetchall()
    labels = [tag[0] for tag in tags_from_db]

    result = classifier(text, candidate_labels=labels)
    print(result)
    tags = [tag for tag, score in zip(result['labels'], result['scores']) if score > 0.084]

    # Сохраняем информацию о документе в базе данных
    save_document(request.form['title'], filename, tags)

    # Возвращаем результат
    return jsonify({
        "tags": tags,
        "text": text,
        "file_path": filename
    })

if __name__ == "__main__":
    app.run(port=5000)
