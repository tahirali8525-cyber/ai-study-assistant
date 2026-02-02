from fastapi import FastAPI, File, UploadFile, Form
from transformers import pipeline
from PIL import Image
import pytesseract
import io

app = FastAPI(title="AI Study Assistant")

# Small AI models for free-tier deployment
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

@app.post("/summarize")
async def summarize_text(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")
    summary = summarizer(text, max_length=150, min_length=50)[0]['summary_text']
    flashcards = [sentence.strip() for sentence in text.split('.')[:5] if sentence]
    return {"summary": summary, "flashcards": flashcards}

@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(io.BytesIO(content))
    text = pytesseract.image_to_string(image)
    summary = summarizer(text, max_length=100)[0]['summary_text']
    return {"text": text, "summary": summary}

@app.post("/chat")
async def chat_query(query: str = Form(...), context: str = Form("")):
    if context:
        answer = qa_model(question=query, context=context)['answer']
    else:
        answer = "Please provide context from a document."
    return {"response": answer}

@app.get("/planner")
def get_planner():
    return {"goals": [{"text": "Study Math", "deadline": "2023-10-01"},
                      {"text": "Finish Project", "deadline": "2023-10-05"}]}

@app.post("/add_goal")
async def add_goal(text: str = Form(...), deadline: str = Form(...)):
    return {"message": f"Goal '{text}' added with deadline {deadline}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
