# sentence_generator.py
import os
from openai import OpenAI

client = OpenAI(
    api_key="sk-jnxczqwnnvdclijbajxylxrfgwbtopweipfywcsgcsqdnpxn", 
    base_url="https://api.siliconflow.cn/v1/",
)

def generate_sentence_from_letters(letters, system_prompt=None):
    """
    letters: list of single-character strings, e.g. ['H','E','L','L','O']
    system_prompt: 可选，对助手的身份或风格进行说明
    返回：生成的完整、有意义的句子
    """
    if not letters:
        return ""

    # 默认的 system prompt，告诉模型要做什么
    if system_prompt is None:
        system_prompt = (
            "You are a post-processing engine for a sign language recognition API. You will receive a raw string of letters output by the camera, which may contain omissions, duplicates, or misrecognized characters. Your job is to infer the single most likely meaningful sentence or word that the user intended. Output only that sentence or word, and nothing else. Remember: \n\n1. The letters “j” and “z” are never provided by the API; if they belong in the intended sentence, you must insert them in the correct positions. \n2. Certain letters are easily confused in sign language (for example, a, e, m, n, s), so you must correct likely substitutions. \n3. Do not output any commentary, explanation, or extra characters—only the reconstructed sentence. "
        )

    user_input = "".join(letters)
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"：{user_input}"},
    ]

    response = client.chat.completions.create(
            model = 'Qwen/Qwen3-8B',
            messages=prompt,
            temperature=0.0,
        )
    return response.choices[0].message.content.strip()
