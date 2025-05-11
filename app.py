import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template_string, request, send_file
from analyzer import analyze_text
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import base64
from collections import Counter
import re

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Анализ экстремистского контента</title>
</head>
<body>
    <h2>Проверка поста на экстремизм</h2>
    <form method="post">
        <label>Введите текст вручную:</label><br>
        <textarea name="text" rows="5" cols="80" placeholder="Введите текст...">{{ default_text }}</textarea><br><br>

        <input type="submit" value="Анализировать">
    </form>

    {% if result %}
        <h3>Результат анализа:</h3>
        <p><b>Вероятность экстремистского контента:</b> {{ probability }}%</p>
        <p><b>Классификация:</b> {{ label }}</p>

        <h4>Визуализация вероятности:</h4>
        <img src="data:image/png;base64,{{ prob_chart }}"/><br><br>

        <h4>Облако слов (введённый текст):</h4>
        <img src="data:image/png;base64,{{ wordcloud }}"/><br><br>

        <h4>Частотный анализ (топ-слов):</h4>
        <img src="data:image/png;base64,{{ freq_chart }}"/><br><br>

        <h4>Оценка по предложениям:</h4>
        <img src="data:image/png;base64,{{ sentence_chart }}"/>
    {% endif %}
</body>
</html>
"""

def generate_probability_chart(prob):
    fig, ax = plt.subplots()
    ax.bar(['Нейтральный', 'Экстремизм'], [1 - prob, prob], color=['green', 'red'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Вероятность')
    ax.set_title('Распределение вероятности')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded

def generate_wordcloud(text):
    wc = WordCloud(width=600, height=400, background_color='white').generate(text)
    buf = io.BytesIO()
    wc.to_image().save(buf, format='PNG')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    return encoded

def generate_freq_chart(text):
    words = re.findall(r"\w{3,}", text.lower())
    common = Counter(words).most_common(5)
    fig, ax = plt.subplots()
    ax.bar([w for w, _ in common], [c for _, c in common], color='blue')
    ax.set_title("Топ-5 слов по частоте")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded

def generate_sentence_chart(text):
    from analyzer import analyze_text  # безопасный импорт внутри функции
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 5]
    probs = [analyze_text(sent)['probability'] / 100 for sent in sentences]
    fig, ax = plt.subplots()
    ax.plot(range(1, len(probs)+1), probs, marker='o')
    ax.set_title("Оценка по предложениям")
    ax.set_xlabel("Предложение")
    ax.set_ylabel("Вероятность экстремизма")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None
    label = None
    default_text = ""
    default_link = ""
    prob_chart = ""
    wordcloud = ""
    freq_chart = ""
    sentence_chart = ""

    if request.method == "POST":
        text = request.form.get("text", "").strip()
        default_text = text

        if text:
            result_data = analyze_text(text)
            probability = result_data["probability"]
            print(f"TEST PROB: {probability}")
            label = result_data["label"]
            prob_chart = generate_probability_chart(probability / 100)
            wordcloud = generate_wordcloud(text)
            freq_chart = generate_freq_chart(text)
            sentence_chart = generate_sentence_chart(text)
            result = True

    return render_template_string(HTML_TEMPLATE, result=result, probability=probability, label=label,
                                  default_text=default_text, default_link=default_link,
                                  prob_chart=prob_chart, wordcloud=wordcloud,
                                  freq_chart=freq_chart, sentence_chart=sentence_chart)

if __name__ == "__main__":
    app.run(debug=True)
