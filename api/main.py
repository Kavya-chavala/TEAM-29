from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from openai import OpenAI
import os
from dotenv import load_dotenv

from api.fetch_label import fetch_drug_label
from api.rag import create_temp_vector_store, retrieve_answer
from api.reminder import generate_custom_reminder
from openai import RateLimitError

load_dotenv()

app = FastAPI(
    title="Medication Assistant",
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- HOME UI ----------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Medication Assistant</title>
    <style>
        body {
            font-family: Arial;
            background: #f2f4f7;
            padding: 40px;
        }
        .card {
            background: white;
            padding: 25px;
            max-width: 600px;
            margin: auto;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        h2 { color: #2c3e50; }
        input, button {
            width: 100%;
            padding: 10px;
            margin-top: 10px;
        }
        button {
            background: #3498db;
            color: white;
            border: none;
            border-radius: 6px;
        }
        .section {
            margin-top: 30px;
        }
    </style>
</head>
<body>

<div class="card">
    <h2>💊 Drug Information</h2>
    <form action="/drug-info" method="post">
        <input name="drug" placeholder="Enter medicine name" required>
        <input name="question" placeholder="Ask about dosage, side effects..." required>
        <button type="submit">Get Information</button>
    </form>

    <div class="section">
        <h2>⏰ Medication Reminder</h2>
        <form action="/reminder-ui" method="post">
            <input name="medicine" placeholder="Medicine name" required>
            <input name="dose" placeholder="Dose (e.g. 500mg)" required>
            <input name="frequency" placeholder="Frequency (e.g. 3 times/day)" required>
            <input name="times" placeholder="Times (8am,2pm,8pm)" required>
            <button type="submit">Create Reminder</button>
        </form>
    </div>
</div>

<script>
function scheduleNotifications(medicine, dose, times) {

    if (Notification.permission !== "granted") {
        Notification.requestPermission();
    }

    times.forEach(time => {
        let now = new Date();
        let reminderTime = new Date();

        let parts = time.match(/(\d+)(am|pm)/i);
        if (!parts) return;

        let hour = parseInt(parts[1]);
        let period = parts[2].toLowerCase();

        if (period === "pm" && hour !== 12) hour += 12;
        if (period === "am" && hour === 12) hour = 0;

        reminderTime.setHours(hour, 0, 0, 0);

        if (reminderTime <= now) {
            reminderTime.setDate(reminderTime.getDate() + 1);
        }

        let timeout = reminderTime - now;

        setTimeout(() => {
            new Notification("Medication Reminder", {
                body: `Time to take ${medicine} — ${dose}`,
                icon: "https://cdn-icons-png.flaticon.com/512/2965/2965567.png"
            });
        }, timeout);
    });
}
</script>

</body>
</html>
"""

# ---------------- DRUG INFO RESULT ----------------
@app.post("/drug-info", response_class=HTMLResponse)
def drug_info(drug: str = Form(...), question: str = Form(...)):

    label_text = fetch_drug_label(drug)

    if not label_text or label_text.strip() == "":
        return """
        <h3>No drug information found.</h3>
        <a href="/">⬅ Back</a>
        """

    try:
        collection = create_temp_vector_store(label_text)
        context = retrieve_answer(collection, question)
    except Exception as e:
        return f"<p>Error in RAG pipeline: {str(e)}</p><a href='/'>⬅ Back</a>"

    prompt = f"""
Use ONLY the following drug label information:

{context}

Question: {question}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        # ✅ FIXED VERSION HERE
        answer = response.choices[0].message.content

    except RateLimitError:
        answer = "⚠️ API quota exceeded. Please wait or use another key."

    except Exception as e:
        answer = f"⚠️ Error: {str(e)}"

    return f"""
        <h3>Answer</h3>
        <p>{answer}</p>
        <a href="/">⬅ Back</a>
    """

# ---------------- REMINDER RESULT ----------------
@app.post("/reminder-ui", response_class=HTMLResponse)
def reminder_ui(
    medicine: str = Form(...),
    dose: str = Form(...),
    frequency: str = Form(...),
    times: str = Form(...)
):
    times_list = [t.strip() for t in times.split(",")]
    reminder = generate_custom_reminder(medicine, dose, frequency, times_list)

    return f"""
<h3>Medication Reminder Set Successfully</h3>

<p><b>Medicine:</b> {reminder['medicine']}</p>
<p><b>Dose:</b> {reminder['dose']}</p>
<p><b>Frequency:</b> {reminder['frequency']}</p>
<p><b>Times:</b> {', '.join(reminder['reminder_times'])}</p>

<script>
    const medicine = "{reminder['medicine']}";
    const dose = "{reminder['dose']}";
    const times = "{', '.join(reminder['reminder_times'])}".split(",");

    const cleanTimes = times.map(t => t.trim());
    scheduleNotifications(medicine, dose, cleanTimes);
</script>

<a href="/">⬅ Back</a>
"""
