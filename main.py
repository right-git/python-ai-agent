import uvicorn
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from loguru import logger
from openai import AsyncOpenAI

from config import HELICONE_API_KEY, SYSTEM_PROMPT, TOOLS
from functions import run_command, save_code, search, fetch_page

app = FastAPI()

openai_client = AsyncOpenAI(
    base_url="https://oai.helicone.ai/v1",
    default_headers={
        "Helicone-Auth": f"Bearer {HELICONE_API_KEY}"
    }
)


@app.get("/")
async def index():
    with open("index.html", "r", encoding="UTF-8") as f:
        html = f.read()
        
    return HTMLResponse(html)



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    chat_history = [{
        "role": "system",
        "content": SYSTEM_PROMPT
    }]
    
    try:
        while True:
            data_frontend = await websocket.receive_text()
            user_input = data_frontend.strip()
            logger.debug(f"Сообщение от фронта: {user_input}")
            
            chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            while True:        
                ai_response = await openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=chat_history,
                    tools=TOOLS,
                    tool_choice="auto",
                    parallel_tool_calls=True
                )
            
                ai_message = ai_response.choices[0].message
                
                chat_history.append(ai_message.to_dict()) 

                if not ai_message.tool_calls:
                    await websocket.send_text(json.dumps({
                        "role": "assistant",
                        "content": ai_message.content
                    }))
                    break
                
                
                for tool_call in ai_message.tool_calls:
                    func_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    
                    result = ""
                    
                    try:
                        if func_name == "run_command":
                            if "input_str" in args: 
                                result = run_command(args["command"], args["input_str"])
                            else:
                                result = run_command(args["command"])

                        elif func_name == "save_code":
                            result = save_code(args["code"], args["filename"])
                        elif func_name == "search":
                            result = search(args["query"])
                        elif func_name == "fetch_page":
                            result = await fetch_page(args["url"])
                        else:
                            result = f"Неизвестная функция {func_name}"
                    except Exception as e:
                        result = f"Ошибка вызова функции {func_name} {str(e)}"
                

                    chat_history.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tool_call.id
                    })
            
            
    except WebSocketDisconnect:
        logger.error("Клиент отсоединил соединение")
        



if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=False)
