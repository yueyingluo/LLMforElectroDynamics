from openai import OpenAI
import json

api_base = "http://127.0.0.1:6006/v1"
api_key = "dummy_kep"
model_name = "/data/group_038/qwen2.5-14B-ref"
problem_set = "test_problem.json"
group_num = "038"

client = OpenAI(
    base_url=api_base,
    api_key=api_key,
)

def predict(message):
    response = client.chat.completions.create(
        model= model_name,
        messages=[
            {'role': 'user', 'content': message+"Reason step by step and provide a clear Final Answer."},],
        max_tokens=8196,
        temperature=0.3,
        stream=False
    )
    return response.choices[0].message.content


if __name__=="__main__":
    with open(problem_set, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for i in range(len(data)):
        problem = data[i]["problem"]
        print(f"Working on problem {i}.")
        answer = predict(problem)
        print(f"problem {i} done!")
        data[i]["answer"] = answer

    with open(f'Answer_{group_num}.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

