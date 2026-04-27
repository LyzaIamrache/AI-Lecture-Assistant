from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import time

start_time = time.time()

print("\nSTEP 1: AUDIO INPUT")
print("Input file: audio.wav")

print("\nSTEP 2: SPEECH-TO-TEXT TRANSCRIPTION")
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")

result = asr("audio.wav")
transcription = result["text"].strip()

print("\nTRANSCRIPTION:")
print(transcription)

print("\nSTEP 3: TEXT PROCESSING")
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_text(prompt, max_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

summary = generate_text(
    "Create a clear academic summary of this lecture in 3 sentences. "
    "Include the main concept, the example, and why the concept is useful:\n"
    + transcription,
    180
)

key_points = generate_text(
    "Extract 4 specific bullet-point key ideas from this lecture. "
    "Use information only from the lecture text:\n"
    + transcription,
    180
)

study_questions = generate_text(
    "Create 4 study questions based only on this lecture. "
    "The questions should help students review the main ideas and example:\n"
    + transcription,
    180
)

print("\nSTEP 4: SUMMARY")
print(summary)

print("\nSTEP 5: KEY POINTS")
print(key_points)

print("\nSTEP 6: STUDY QUESTIONS")
print(study_questions)

end_time = time.time()
latency = round(end_time - start_time, 2)

print("\nSTEP 7: LATENCY")
print(latency, "seconds")

with open("output.txt", "w", encoding="utf-8") as f:
    f.write("AI Lecture Support Assistant Output\n\n")
    f.write("STEP 1: AUDIO INPUT\n")
    f.write("Input file: audio.wav\n\n")

    f.write("STEP 2: SPEECH-TO-TEXT TRANSCRIPTION\n")
    f.write(transcription + "\n\n")

    f.write("STEP 3: SUMMARY\n")
    f.write(summary + "\n\n")

    f.write("STEP 4: KEY POINTS\n")
    f.write(key_points + "\n\n")

    f.write("STEP 5: STUDY QUESTIONS\n")
    f.write(study_questions + "\n\n")

    f.write("STEP 6: LATENCY\n")
    f.write(str(latency) + " seconds\n")
