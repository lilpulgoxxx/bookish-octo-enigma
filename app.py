from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from vllm import LLM
from vllm.sampling_params import SamplingParams
from huggingface_hub import hf_hub_download
from datetime import datetime, timedelta
import io

# Configuración inicial
model_name = "xfcxcxcdfdfd/1-bit"
model_display_name = "Cyrah"  # Nombre automático del modelo
app = FastAPI()

# Función para generar el sistema de *prompt* de forma automática
def generate_system_prompt(repo_id: str) -> str:
    today = datetime.today().strftime('%Y-%m-%d')
    yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    model_name = repo_id.split("/")[-1]
    system_prompt = f"""
    You are {model_display_name}, an AI trained to assist users.
    Today's date is {today}, and yesterday was {yesterday}.
    Your task is to provide helpful responses based on the user inputs.
    If asked, you will respond with your name as "{model_display_name}".
    """
    return system_prompt

# Inicialización del modelo con manejo de errores
def initialize_model(model_name: str):
    try:
        # Intentar cargar el modelo con configuración llama3
        model = LLM(
            model=model_name,
            config_format="llama3",
            load_format="llama3",
            tokenizer_mode="llama3",
            tensor_parallel_size=8,
            limit_mm_per_prompt={}
        )
        print("Modelo cargado con configuración llama3.")
    except Exception as e:
        print(f"Error al cargar modelo con llama3: {e}. Cargando modelo con llama.")
        # Si falla, cargar el modelo con configuración llama
        model = LLM(
            model=model_name,
            config_format="llama",
            load_format="llama",
            tokenizer_mode="llama",
            tensor_parallel_size=8,
            limit_mm_per_prompt={}
        )
        print("Modelo cargado con configuración llama.")
    
    return model

# Cargar el modelo
llm = initialize_model(model_name)

sampling_params = SamplingParams(max_tokens=512)

# Función para generar la respuesta, dividiendo en partes si excede los tokens
def generate_long_response(messages, sampling_params):
    full_response = ""
    token_limit = sampling_params.max_tokens

    # Bucle para manejar la división de respuestas
    while True:
        outputs = llm.chat(messages, sampling_params=sampling_params)
        response_text = outputs[0].outputs[0].text
        
        # Concatenamos la respuesta
        full_response += response_text

        # Si la respuesta supera el límite de tokens, dividimos
        if len(full_response.split()) > token_limit:
            # Dividir la respuesta y continuar donde se quedó
            messages[1]["content"] = response_text.split()[-token_limit:]
            continue
        else:
            break

    return full_response

# Endpoint para recibir la pregunta como texto en el cuerpo de la solicitud y retornar en streaming
@app.post("/generate/")
async def generate_response(request: Request):
    user_message = await request.body()
    user_message = user_message.decode('utf-8')

    # Generar el prompt del sistema automáticamente
    system_prompt = generate_system_prompt(model_name)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    
    def generate_streaming_response(messages, sampling_params):
        try:
            full_response = ""
            token_limit = sampling_params.max_tokens
            while True:
                outputs = llm.chat(messages, sampling_params=sampling_params)
                response_text = outputs[0].outputs[0].text
                
                # Concatenamos la respuesta
                full_response += response_text

                # Si la respuesta supera el límite de tokens, dividimos y continuamos
                yield response_text  # Aquí se manda la parte generada como un 'stream'
                
                if len(full_response.split()) > token_limit:
                    # Dividir la respuesta y continuar donde se quedó
                    messages[1]["content"] = response_text.split()[-token_limit:]
                    continue
                else:
                    break

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Devolvemos la respuesta en streaming
    return StreamingResponse(generate_streaming_response(messages, sampling_params), media_type="text/plain")


# Inicio del servidor con Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
