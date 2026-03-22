import os 
import json 
import uuid 
import random 
import string 
from dotenv import load_dotenv 
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, HTTPException, Depends 
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm 
from pydantic import BaseModel 
from typing import Any, List, Dict, Optional
import asyncpg 
from passlib.context import CryptContext 
import jwt 
from datetime import datetime, timedelta 
 
from langchain_openai import ChatOpenAI 
from langchain_openai import OpenAIEmbeddings 
from langchain.chains.question_answering import load_qa_chain 
from fastapi.middleware.cors import CORSMiddleware 
 
import vercel_blob 
 
load_dotenv() 
 
app = FastAPI() 
 
app.add_middleware( 
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"], 
) 
 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
DATABASE_URL = "your-postgres-url" 
SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key") 
ALGORITHM = "HS256" 
ACCESS_TOKEN_EXPIRE_MINUTES = 60 
 
chat = ChatOpenAI(model="gpt-4o", temperature=0) 
embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY) 
 
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto") 
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token") 
 
questions_list = [ 
    "What is your age?", 
    "What is your gender?", 
    "What is your marital status?", 
    "What is your employment status?", 
    "Do you have any dependents?", 
    "How many cigarettes do you smoke per day, if any?", 
    "How often do you consume alcoholic beverages?", 
    "Describe your typical weekly physical activity (type, duration, intensity).", 
    "What is your usual dietary pattern? (e.g., vegetarian, high-protein, etc.)", 
    "Do you use tobacco products? If yes, specify the type and frequency.", 
    "Over the last month, how often have you felt stressed or overwhelmed?", 
    "Do you have a history of depression or anxiety? If yes, are you currently receiving treatment?", 
    "How often do you feel socially connected with friends, family, or community?", 
    "How many hours of sleep do you get on an average night?", 
    "Have you experienced any recent changes in your weight?", 
    "Do you have any chronic conditions (e.g., diabetes, hypertension)?", 
    "When was your last medical check-up?", 
    "Are you currently taking any prescription medications? If yes, please list them.", 
    "Have you had any surgeries or hospitalizations in the past year?", 
    "Have you had any previous cancer diagnoses? If yes, please specify the type and treatment received.", 
    "Have you undergone any radiation therapy or chemotherapy in the past?", 
    "Do you have a history of significant sunburns or tanning bed usage?", 
    "Have you experienced any symptoms consistent with dengue fever in the past year (e.g., high fever, severe headaches, pain behind the eyes, joint and muscle pain, fatigue, nausea, vomiting, skin rash, or mild bleeding)?", 
    "Have you been diagnosed with dengue in the past? If so, when and what was the severity?", 
    "Are you up to date with your vaccinations (e.g., influenza, tetanus)?", 
    "When was your last dental check-up?", 
    "Have you had any cancer screenings? (e.g., mammogram, colonoscopy)", 
    "Is there a history of any chronic diseases in your family (e.g., heart disease, stroke, cancer)?", 
    "Is there a history of cancer in your family? If yes, specify the types of cancer and the relatives affected.", 
    "Have any family members been genetically tested for cancer markers? If yes, what were the findings?", 
    "Are you currently pregnant or planning to become pregnant in the near future?", 
    "When was your last gynecological exam?", 
    "How often do you perform a breast self-exam?", 
    "Have you ever been diagnosed with HPV or any other sexually transmitted infections known to increase cancer risk?", 
    "Do you use hormone replacement therapy?", 
    "Have you had a prostate exam?", 
    "Do you experience any issues related to urinary function?", 
    "Have you had a prostate-specific antigen (PSA) test? If so, when?", 
    "For men: Do you regularly examine yourself for testicular abnormalities?" 
] 
 
class QueryResponse(BaseModel): 
    matched_questions_answers: List[Dict[str, str]] 
 
@app.get("/") 
def read_root(): 
    return {"Hello": "World"} 
 
@app.get("/documents") 
async def get_pdf_vercel(): 
    return vercel_blob.list() 
 
@app.post("/upload") 
async def upload_vercel(file: UploadFile = File(...)): 
    vercel_blob.put(file.filename, await file.read(), {}) 
    return {"File Uploaded"} 
 
class QueryRequest(BaseModel): 
    text: str 
 
class Document: 
    def __init__(self, page_content: str, metadata: Any = None): 
        self.page_content = page_content 
        self.metadata = metadata or {} 
 
async def init_db(): 
    conn = await asyncpg.connect(DATABASE_URL) 
    await conn.execute(''' 
        CREATE TABLE IF NOT EXISTS q_and_a ( 
            id SERIAL PRIMARY KEY, 
            data TEXT 
        ); 
        CREATE TABLE IF NOT EXISTS report ( 
            id SERIAL PRIMARY KEY, 
            data TEXT 
        ); 
        CREATE TABLE IF NOT EXISTS transcript ( 
            id SERIAL PRIMARY KEY, 
            data TEXT 
        ); 
        CREATE TABLE IF NOT EXISTS users ( 
            id UUID PRIMARY KEY, 
            email TEXT UNIQUE NOT NULL, 
            hashed_password TEXT NOT NULL, 
            role TEXT NOT NULL 
        ); 
        CREATE TABLE IF NOT EXISTS patients ( 
            patient_id UUID PRIMARY KEY, 
            name TEXT, 
            address TEXT, 
            phone_number TEXT, 
            date_of_birth DATE, 
            profile_picture_url TEXT, 
            diet_plan TEXT, 
            FOREIGN KEY (patient_id) REFERENCES users (id) 
        ); 
        CREATE TABLE IF NOT EXISTS clinicians ( 
            clinician_id UUID PRIMARY KEY, 
            name TEXT, 
            email TEXT, 
            date_of_birth DATE, 
            contact_number TEXT, 
            address TEXT, 
            FOREIGN KEY (clinician_id) REFERENCES users (id) 
        ); 
        CREATE TABLE IF NOT EXISTS meeting ( 
            meeting_id UUID PRIMARY KEY, 
            user_id UUID REFERENCES users(id), 
            meeting_datetime DATE, 
            report_link TEXT, 
            transcript TEXT, 
            q_and_a TEXT 
        ); 
    ''') 
    await conn.close() 
 
@app.on_event("startup") 
async def startup(): 
    await init_db() 
 
@app.post("/query") 
async def query(request: QueryRequest): 
    text = request.text 
     
    matched_questions_answers = [] 
    for question in questions_list: 
        # Performing similarity search for each question 
        match = Document(page_content=text) 
        if match: 
            # Loads the LLM chain 
            chain = load_qa_chain(chat, chain_type="stuff") 
            # Adding contextual information to the prompt 
            context = ( 
                "This is a conversation between a clinician and a patient. The clinician is asking questions to gather information about the patient's health and lifestyle. " 
                "If there is not enough information even after checking the context, then respond with 'No Response' but remember to check the context. " 
                "Please provide detailed and relevant answers based on the context provided." 
                "Also answer should be in third person" 
            ) 
            # Generating the response based on input document 
            response = chain.run(input_documents=[match], question=f"{context}\n\n{question}") 
            # Check if the response contains meaningful information 
            if "No Response" not in response and "No response" not in response: 
                matched_questions_answers.append({"question": question, "answer": response}) 
 
    serialized_data = json.dumps({"matched_questions_answers": matched_questions_answers}) 
    print(f"Serialized data for /query: {serialized_data}")  # Debugging log 
 
    conn = await asyncpg.connect(DATABASE_URL) 
    await conn.execute(''' 
        INSERT INTO q_and_a (data) VALUES ($1) 
    ''', serialized_data) 
    await conn.close() 
 
    return {"matched_questions_answers": matched_questions_answers} 
 
@app.get("/questions") 
async def get_questions(): 
    return {"questions": questions_list} 
 
@app.post("/summarize") 
async def summarize(query_response: QueryResponse): 
    # Convert the list of questions and answers to a string format 
    qa_pairs = "\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in query_response.matched_questions_answers]) 
     
    # Create a prompt for the LLM to categorize and summarize the questions and answers 
    prompt = ( 
        f"Here are some questions and answers from a conversation between a clinician and a patient:\n\n{qa_pairs}\n\n" 
        "Please categorize the questions and answers into relevant categories such as 'Personal Information', 'Medical History', 'Physical Activity', 'Dietary Pattern', 'Smoking', 'Drinking'" 
        "and add more category on your own, if information about certain category is not available then do not send it in response" 
        "For each category, provide a brief summary of the key points discussed." 
        "Your response should in the following manner: Category: Category Name followed by category description. 'Category:' should be present in front of every category heading and content should not start with hyphen(-). there should be a category for summary and a different category for action items" 
        "Also in the end give a summary and action items based on the conversation of the clinician and patient separately as different category" 
    ) 
 
    # Generate the categorized summary using the LLM 
    chain = load_qa_chain(chat, chain_type="stuff") 
    categorized_summary = chain.run(input_documents=[], question=prompt) 
 
    # Parse the categorized summary into a structured format 
    categories = {} 
    current_category = None 
    for line in categorized_summary.split("\n"): 
        if line.startswith("Category:"): 
            current_category = line.replace("Category:", "").strip() 
            categories[current_category] = [] 
        elif current_category and line.strip(): 
            categories[current_category].append(line.strip()) 
 
    # Format the response 
    response = [{"category": category, "content": " ".join(content)} for category, content in categories.items()] 
 
    serialized_data = json.dumps({"categorized_summary": response}) 
    print(f"Serialized data for /summarize: {serialized_data}")  # Debugging log 
 
    conn = await asyncpg.connect(DATABASE_URL) 
    await conn.execute(''' 
        INSERT INTO report (data) VALUES ($1) 
    ''', serialized_data) 
    await conn.close() 
 
    return {"categorized_summary": response} 
 
@app.websocket("/chatWebSocket") 
async def websocket_endpoint(websocket: WebSocket): 
    await websocket.accept()     
    try: 
        while True: 
            data = await websocket.receive_text() 
            print(data) 
            request = QueryRequest(text=data) 
            print(request) 
            response = await query(request) 
            print(response) 
            await websocket.send_json(response) 
 
            conn = await asyncpg.connect(DATABASE_URL) 
            await conn.execute(''' 
                INSERT INTO transcript (data) VALUES ($1) 
            ''', data) 
            await conn.close() 
    except WebSocketDisconnect: 
        print("Client disconnected") 
 
class User(BaseModel): 
    email: str 
    password: str 
 
class UserId(BaseModel): 
    user_id: str 
 
class Token(BaseModel): 
    access_token: str 
    token_type: str 
    user_id: str 
    role: str 
 
class PatientDetails(BaseModel): 
    patient_id: str 
    name: str 
    address: str 
    phone_number: str 
    date_of_birth: datetime 
    profile_picture_url: str 
    diet_plan: str 
 
class UserResponse(BaseModel): 
    id: str 
    email: str 
    role: str 
 
class ClinicianDetails(BaseModel): 
    name: str 
    email: str 
    date_of_birth: datetime 
    contact_number: str 
    address: str 
 
class ClinicianResponse(BaseModel): 
    clinician_id: str 
    name: str 
    email: str 
    date_of_birth: datetime 
    contact_number: str 
    address: str 
    password: str 
 
class MeetingDetails(BaseModel): 
    user_id: str 
    meeting_datetime: datetime 
 
class MeetingUpdate(BaseModel): 
    meeting_id: str 
    report_link: str 
    transcript: str 
    q_and_a: str 
 
class MeetingResponse(BaseModel): 
    meeting_id: str 
    user_id: str 
    meeting_datetime: datetime 
    report_link: Optional[str] = None
    transcript: Optional[str] = None
    q_and_a: Optional[str] = None
 
def verify_password(plain_password, hashed_password): 
    return pwd_context.verify(plain_password, hashed_password) 
 
def get_password_hash(password): 
    return pwd_context.hash(password) 
 
def create_access_token(data: dict, expires_delta: timedelta = None): 
    to_encode = data.copy() 
    if expires_delta: 
        expire = datetime.utcnow() + expires_delta 
    else: 
        expire = datetime.utcnow() + timedelta(minutes=60) 
    to_encode.update({"exp": expire}) 
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM) 
    return encoded_jwt 
 
def generate_password(length: int = 12): 
    characters = string.ascii_letters + string.digits + string.punctuation 
    password = ''.join(random.choice(characters) for i in range(length)) 
    return password 
 
async def get_user(email: str): 
    conn = await asyncpg.connect(DATABASE_URL) 
    user = await conn.fetchrow('SELECT * FROM users WHERE email = $1', email) 
    await conn.close() 
    return user 
 
async def authenticate_user(email: str, password: str): 
    user = await get_user(email) 
    if not user: 
        return False 
    if not verify_password(password, user['hashed_password']): 
        return False 
    return user 
 
@app.post("/register", response_model=UserId) 
async def register(user: User): 
    user_data = await get_user(user.email) 
    if user_data: 
        raise HTTPException(status_code=400, detail="Email already registered") 
     
    hashed_password = get_password_hash(user.password) 
    user_id = str(uuid.uuid4()) 
    conn = await asyncpg.connect(DATABASE_URL) 
    await conn.execute(''' 
        INSERT INTO users (id, email, hashed_password, role) VALUES ($1, $2, $3, $4) 
    ''', user_id, user.email, hashed_password, 'patient') 
    await conn.close() 
 
    return {"user_id" : user_id} 
 
@app.post("/registerClinician", response_model=ClinicianResponse) 
async def register_clinician(details: ClinicianDetails): 
    user_data = await get_user(details.email) 
    if user_data: 
        raise HTTPException(status_code=400, detail="Email already registered") 
     
    password = generate_password() 
    hashed_password = get_password_hash(password) 
    user_id = str(uuid.uuid4()) 
    conn = await asyncpg.connect(DATABASE_URL) 
    await conn.execute(''' 
        INSERT INTO users (id, email, hashed_password, role) VALUES ($1, $2, $3, $4) 
    ''', user_id, details.email, hashed_password, 'clinician') 
    await conn.execute(''' 
        INSERT INTO clinicians (clinician_id, name, email, date_of_birth, contact_number, address)  
        VALUES ($1, $2, $3, $4, $5, $6) 
    ''', user_id, details.name, details.email, details.date_of_birth, details.contact_number, details.address) 
    await conn.close() 
 
    return { 
        "clinician_id": user_id, 
        "name": details.name, 
        "email": details.email, 
        "date_of_birth": details.date_of_birth, 
        "contact_number": details.contact_number, 
        "address": details.address, 
        "password": password 
    } 
 
@app.post("/login", response_model=Token) 
async def login(form_data: OAuth2PasswordRequestForm = Depends()): 
    user = await authenticate_user(form_data.username, form_data.password) 
    if not user: 
        raise HTTPException( 
            status_code=400, 
            detail="Incorrect email or password", 
            headers={"WWW-Authenticate": "Bearer"}, 
        ) 
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES) 
    access_token = create_access_token( 
        data={"sub": user['email']}, expires_delta=access_token_expires 
    ) 
    return {"user_id": str(user['id']), "role": str(user['role']), "access_token": access_token, "token_type": "bearer"} 
 
@app.post("/patients", response_model=PatientDetails) 
async def create_patient_details(details: PatientDetails): 
    conn = await asyncpg.connect(DATABASE_URL) 
    await conn.execute(''' 
        INSERT INTO patients (patient_id, name, address, phone_number, date_of_birth, profile_picture_url, diet_plan)  
        VALUES ($1, $2, $3, $4, $5, $6, $7) 
    ''', details.patient_id, details.name, details.address, details.phone_number, details.date_of_birth, details.profile_picture_url, details.diet_plan) 
    await conn.close() 
    return details 
 
@app.get("/patients/{patient_id}", response_model=PatientDetails) 
async def get_patient_details(patient_id: str): 
    conn = await asyncpg.connect(DATABASE_URL) 
    patient = await conn.fetchrow('SELECT * FROM patients WHERE patient_id = $1', patient_id) 
    await conn.close() 
    if not patient: 
        raise HTTPException(status_code=404, detail="Patient not found") 
    patient_dict = dict(patient) 
    patient_dict['patient_id'] = str(patient_dict['patient_id'])  # Convert UUID to string
    return patient_dict 
 
@app.get("/users/{user_id}", response_model=UserResponse) 
async def get_user_details(user_id: str): 
    conn = await asyncpg.connect(DATABASE_URL) 
    user = await conn.fetchrow('SELECT * FROM users WHERE id = $1', user_id) 
    await conn.close() 
    if not user: 
        raise HTTPException(status_code=404, detail="User not found") 
    user_dict = dict(user) 
    user_dict['id'] = str(user_dict['id'])  # Convert UUID to string 
    return user_dict 
 
@app.get("/clinicians/{clinician_id}", response_model=ClinicianDetails) 
async def get_clinician_details(clinician_id: str): 
    conn = await asyncpg.connect(DATABASE_URL) 
    clinician = await conn.fetchrow('SELECT * FROM clinicians WHERE clinician_id = $1', clinician_id) 
    await conn.close() 
    if not clinician: 
        raise HTTPException(status_code=404, detail="Clinician not found") 
    clinician_dict = dict(clinician) 
    clinician_dict['clinician_id'] = str(clinician_dict['clinician_id'])  # Convert UUID to string 
    return clinician_dict

@app.post("/meetings", response_model=MeetingResponse)
async def create_meeting(details: MeetingDetails):
    meeting_id = str(uuid.uuid4())
    conn = await asyncpg.connect(DATABASE_URL)
    await conn.execute('''
        INSERT INTO meeting (meeting_id, user_id, meeting_datetime, report_link, transcript, q_and_a) 
        VALUES ($1, $2, $3, $4, $5, $6)
    ''', meeting_id, details.user_id, details.meeting_datetime, None, None, None)
    await conn.close()
    return {
        "meeting_id": meeting_id,
        "user_id": details.user_id,
        "meeting_datetime": details.meeting_datetime,
        "report_link": "null",
        "transcript": "null",
        "q_and_a": "null"
    }

@app.get("/meetings/{meeting_id}", response_model=MeetingResponse)
async def get_meeting_details(meeting_id: str):
    conn = await asyncpg.connect(DATABASE_URL)
    meeting = await conn.fetchrow('SELECT * FROM meeting WHERE meeting_id = $1', meeting_id)
    await conn.close()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    meeting_dict = dict(meeting)
    meeting_dict['meeting_id'] = str(meeting_dict['meeting_id'])  # Convert UUID to string
    meeting_dict['user_id'] = str(meeting_dict['user_id'])  # Convert UUID to string
    return meeting_dict

@app.put("/meetings/{meeting_id}", response_model=MeetingResponse)
async def update_meeting(meeting_id: str, update: MeetingUpdate):
    conn = await asyncpg.connect(DATABASE_URL)
    await conn.execute('''
        UPDATE meeting 
        SET report_link = $1, transcript = $2, q_and_a = $3 
        WHERE meeting_id = $4
    ''', update.report_link, update.transcript, update.q_and_a, meeting_id)
    meeting = await conn.fetchrow('SELECT * FROM meeting WHERE meeting_id = $1', meeting_id)
    await conn.close()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    meeting_dict = dict(meeting)
    meeting_dict['meeting_id'] = str(meeting_dict['meeting_id'])  # Convert UUID to string
    meeting_dict['user_id'] = str(meeting_dict['user_id'])  # Convert UUID to string
    return meeting_dict

@app.get("/meetings", response_model=List[MeetingResponse])
async def get_all_meetings():
    conn = await asyncpg.connect(DATABASE_URL)
    meetings = await conn.fetch('SELECT * FROM meeting')
    await conn.close()
    meeting_list = []
    for meeting in meetings:
        meeting_dict = dict(meeting)
        meeting_dict['meeting_id'] = str(meeting_dict['meeting_id'])  # Convert UUID to string
        meeting_dict['user_id'] = str(meeting_dict['user_id'])  # Convert UUID to string
        meeting_list.append(meeting_dict)
    return meeting_list

@app.get("/meetings/user/{user_id}", response_model=List[MeetingResponse])
async def get_meetings_by_user_id(user_id: str):
    conn = await asyncpg.connect(DATABASE_URL)
    meetings = await conn.fetch('SELECT * FROM meeting WHERE user_id = $1', user_id)
    await conn.close()
    meeting_list = []
    for meeting in meetings:
        meeting_dict = dict(meeting)
        meeting_dict['meeting_id'] = str(meeting_dict['meeting_id'])  # Convert UUID to string
        meeting_dict['user_id'] = str(meeting_dict['user_id'])  # Convert UUID to string
        meeting_list.append(meeting_dict)
    return meeting_list
