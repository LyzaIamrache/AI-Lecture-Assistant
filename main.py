from transformers import pipeline
import time

start_time = time.time()

# Speech-to-text
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")

result = asr("audio.wav")
transcription = result["text"]

print("\nTRANSCRIPTION:")
print(transcription)

# Generate study support from the transcription
generator = pipeline("text2text-generation", model="google/flan-t5-small")

summary_prompt = "Summarize this lecture in one short paragraph:\n" + transcription
summary = generator(summary_prompt, max_new_tokens=80)[0]["generated_text"]

key_points_prompt = "Write 3 key points from this lecture:\n" + transcription
key_points = generator(key_points_prompt, max_new_tokens=100)[0]["generated_text"]

questions_prompt = "Write 3 study questions based on this lecture:\n" + transcription
questions = generator(questions_prompt, max_new_tokens=100)[0]["generated_text"]

print("\nSUMMARY:")
print(summary)

print("\nKEY POINTS:")
print(key_points)

print("\nSTUDY QUESTIONS:")
print(questions)

end_time = time.time()
print("\nLATENCY:", round(end_time - start_time, 2), "seconds")
