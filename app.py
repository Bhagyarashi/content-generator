from flask import Flask, request, render_template
from transformers import pipeline, AutoTokenizer

app = Flask(__name__)
text_generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M", device=0)  # Adjust 'device' based on your setup
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        format = request.form.get('format', '')
        about = request.form.get('about', '')

        if not format or not about:
            return render_template('index.html', error='Format and about fields are required.')

        content = generate_marketing_content(format, about)
        return render_template('index.html', content=content)

    return render_template('index.html')

def generate_marketing_content(format, about):
    input_text = f"write a {format} content about {about}"
    output = text_generator(input_text, max_length=50, num_return_sequences=1)

    # Check if output is a list and not empty
    if isinstance(output, list) and output:
        generated_text = output[0].get('generated_text')
        if generated_text:
            return generated_text

    return "Failed to generate marketing content."

if __name__ == '__main__':
    app.run(debug=True)

