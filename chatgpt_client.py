import aiohttp
import json

URL = 'http://localhost:3000/api/chat'
METHOD = 'POST'
HEADERS = {
    'Content-Type': 'application/json',
    'Accept': 'text/event-stream'
}

async def stream_chat(message: str):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=URL,
                headers=HEADERS,
                json={'message': message},
                chunked=True
            ) as response:
                response.raise_for_status()
                buffer = ""
                async for chunk in response.content:
                    buffer += chunk.decode()
                    if buffer.endswith('\n\n'):
                        if buffer.startswith('data: '):
                            data = buffer[6:].strip()
                            try:
                                json_data = json.loads(data)
                                if 'response' in json_data:
                                    yield json_data['response']
                            except json.JSONDecodeError:
                                print(f"Failed to parse JSON: {data}")
                        buffer = ""
                    
    except aiohttp.ClientError as e:
        print(f"HTTP error occurred: {e}")
    except json.JSONDecodeError as e:
        print(f"JSON error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
