from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Montar la carpeta 'static' para servir archivos estáticos como CSS e imágenes
app.mount("/static", StaticFiles(directory="static"), name="static")

# Carpeta donde están los HTML
templates = Jinja2Templates(directory="templates")

# Ruta principal ("/") que muestra la página de inicio con un formulario para ingresar nombre
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Esto redirige al menú con el nombre del usuario
@app.get("/menu", response_class=HTMLResponse)
async def menu(request: Request, nombre: str = ""):
    return templates.TemplateResponse("menu.html", {"request": request, "nombre": nombre})

# Esto redirige a la pagina del chatbot
@app.get("/chatbot", response_class=HTMLResponse)
async def chatbot_page(request: Request):
    return templates.TemplateResponse("chatbot.html", {"request": request})

# Endpoint para recibir preguntas del chatbot
@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    pregunta = data.get("pregunta", "").lower()

    # Simulación de respuesta del chatbot
    if "compost" in pregunta:
        respuesta = "El compost ideal para café incluye cáscara, pulpa y estiércol bien fermentado."
    else:
        respuesta = "Lo siento, ¿puedes reformular tu pregunta sobre compostaje en café?"

    return JSONResponse(content={"respuesta": respuesta})

# Ruta para mostrar información sobre café
@app.get("/info/cafe", response_class=HTMLResponse)
async def info_cafe(request: Request):
    return templates.TemplateResponse("informacion_cafe.html", {"request": request})

# Ruta para mostrar información sobre compostaje
@app.get("/info/compostaje", response_class=HTMLResponse)
async def info_compostaje(request: Request):
    return templates.TemplateResponse("informacion_compostaje.html", {"request": request})