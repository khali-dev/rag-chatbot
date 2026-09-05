from src.llm import generate_answer


question = "Was ist die Hauptstadt der Schweiz?"

context = """
[Quelle 1: test.txt]
Die Hauptstadt der Schweiz ist Bern.
""".strip()

answer = generate_answer(
    question=question,
    context=context,
)

print()
print("Antwort:")
print(answer)   