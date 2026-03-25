import string
import random
def format_messages_for_mistral(messages):
    formatted = []

    for m in messages:
        role = m["role"]
        content = m.get("content", "")

        # 🚨 Convert tool messages → assistant messages
        if role == "tool":
            content = f"[TOOL RESULT]\n{content}"
            role = "tool"

        # remove unsupported fields like "name"
        formatted.append({
            "role": role,
            "content": content
        })

    return formatted


def generate_random_id(length=9):
    characters = string.ascii_letters + string.digits
    random_id = ''.join(random.choice(characters) for _ in range(length))
    return random_id