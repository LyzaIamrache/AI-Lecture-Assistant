from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import time

start_time = time.time()

# Speech-to-text
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")

result = asr("audio.wav")
transcription = result["text"]

print("\nTRANSCRIPTION:")
print(transcription)

# Text generation using FLAN-T5 directly
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_text(prompt, max_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

summary = generate_text(
    "Summarize this lecture in one short paragraph: " + transcription,
    80
)

key_points = generate_text(
    "Write three key points from this lecture: " + transcription,
    100
)

questions = generate_text(
    "Write three study questions based on this lecture: " + transcription,
    100
)

print("\nSUMMARY:")
print(summary)

print("\nKEY POINTS:")
print(key_points)

print("\nSTUDY QUESTIONS:")
print(questions)

end_time = time.time()
print("\nLATENCY:", round(end_time - start_time, 2), "seconds")
