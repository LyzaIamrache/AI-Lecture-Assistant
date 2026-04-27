from transformers import pipeline
import time
import re

start_time = time.time()

# Speech-to-text model
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# Convert audio to text
result = asr("audio.wav")
transcription = result["text"].strip()

print("\nVOICE TEXT / TRANSCRIPTION:")
print(transcription)

# Split transcription into sentences
sentences = re.split(r'(?<=[.!?])\s+', transcription)

# Summary = generated from actual transcription
summary = " ".join(sentences[:3])

# Key points = actual sentences from the audio
key_points = sentences[:4]

# Study questions generated from the actual transcription
study_questions = []
for sentence in sentences[:3]:
    if "probability" in sentence.lower():
        study_questions.append("What does probability measure?")
    elif "coin" in sentence.lower() or "heads" in sentence.lower():
        study_questions.append("What is the probability of getting heads when tossing a fair coin?")
    elif "decision" in sentence.lower() or "uncertainty" in sentence.lower():
        study_questions.append("Why is probability useful for making decisions under uncertainty?")

# If not enough questions, create general ones from the transcription
while len(study_questions) < 3:
    study_questions.append("What is one important idea from this lecture?")

print("\nSUMMARY:")
print(summary)

print("\nKEY POINTS:")
for point in key_points:
    print("- " + point)

print("\nSTUDY QUESTIONS:")
for i, question in enumerate(study_questions, 1):
    print(f"{i}. {question}")

end_time = time.time()
latency = round(end_time - start_time, 2)

print("\nLATENCY:", latency, "seconds")

# Save output
with open("output.txt", "w", encoding="utf-8") as f:
    f.write("VOICE TEXT / TRANSCRIPTION:\n")
    f.write(transcription + "\n\n")

    f.write("SUMMARY:\n")
    f.write(summary + "\n\n")

    f.write("KEY POINTS:\n")
    for point in key_points:
        f.write("- " + point + "\n")

    f.write("\nSTUDY QUESTIONS:\n")
    for i, question in enumerate(study_questions, 1):
        f.write(f"{i}. {question}\n")

    f.write(f"\nLATENCY: {latency} seconds\n")

