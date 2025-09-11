# 使用deepseek r1在gsm8k数据集上生成cot，但是目前没有使用，先放在这里

from openai import OpenAI

client = OpenAI(api_key="sk-fc6b700283ee443f9df92cf8d817a147", base_url="https://api.deepseek.com")
prompt = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""
question = """Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"""
prompt = prompt.format(question = question)
response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {"role": "user", "content": prompt},
    ],
    temperature=0,
    stream=False
)

print(response.choices[0].message.content)