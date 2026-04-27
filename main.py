from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import time

start_time = time.time()

# Speech-to-text
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")

result = asr("audio.wav")
transcription = result["text"].strip()

print("\nTRANSCRIPTION:")
print(transcription)

# Load text model
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_text(prompt, max_tokens=120):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Everything below is generated from the audio transcription
summary = generate_text(
    "Summarize the following lecture clearly:\n" + transcription,
    100
)

key_points = generate_text(
    "Write 3 key points from the following lecture:\n" + transcription,
    120
)

study_questions = generate_text(
    "Write 3 study questions based only on the following lecture:\n" + transcription,
    120
)

print("\nSUMMARY:")
print(summary)

print("\nKEY POINTS:")
print(key_points)

print("\nSTUDY QUESTIONS:")
print(study_questions)

end_time = time.time()
latency = round(end_time - start_time, 2)

print("\nLATENCY:", latency, "seconds")

# Save output
with open("output.txt", "w", encoding="utf-8") as f:
    f.write("TRANSCRIPTION:\n")
    f.write(transcription + "\n\n")

    f.write("SUMMARY:\n")
    f.write(summary + "\n\n")

    f.write("KEY POINTS:\n")
    f.write(key_points + "\n\n")

    f.write("STUDY QUESTIONS:\n")
    f.write(study_questions + "\n\n")

    f.write(f"LATENCY: {latency} seconds\n")
