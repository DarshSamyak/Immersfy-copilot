from flask import Flask, request, jsonify, Response
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_openai import OpenAI
from langchain.chains import ConversationChain

from dotenv import load_dotenv

load_dotenv()

import logging, requests, os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", filename="app.log")
app = Flask(__name__)
openai_api_key = os.environ.get("OPEN_AI_KEY")
llm = OpenAI(temperature=0.7, max_tokens=512, openai_api_key=openai_api_key)
window_memory = ConversationBufferWindowMemory(k=5)
conversation = ConversationChain(llm=llm, memory=window_memory)


def process_image(base64_image, prompt):
    logging.info("Process image method called....")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai_api_key}"}

    payload = {"model": "gpt-4-vision-preview", "messages": [{"role": "assistant", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}], "max_tokens": 300}

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return response.json()


@app.route("/copilot", methods=["POST"])
def copilot():
    try:
        logging.info("Copilot Endpoint Called....")
        prompt = request.json.get("prompt")
        if conversation.memory.buffer == "":
            prompt = "As a director copilot answer this: " + prompt

        b64_image = request.json.get("reference_image", False)

        # LLM Input
        if not b64_image:
            # llm_response = llm.invoke(prompt).content
            llm_response = conversation.predict(input=prompt)
            return {"result": llm_response}, 200

        # Vision Model
        vision_response = process_image(b64_image, prompt).get("choices")[0]["message"]["content"]

        return {"result": vision_response}, 200
    except Exception as e:
        return {"result": "Fail", "Exception": e}, 400


if __name__ == "__main__":
    logging.info("Starting the app....")
    app.run(debug=True)
