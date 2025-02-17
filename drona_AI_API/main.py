from fastapi import FastAPI, HTTPException, requests
from typing import List
from agno.agent import Agent
from pydantic import BaseModel, Field
from agno.models.ollama import Ollama
from agno.vectordb.pgvector import PgVector, SearchType
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.models.groq import Groq
from agno.embedder.google import GeminiEmbedder
from agno.tools.duckduckgo import DuckDuckGoTools
from keybert import KeyBERT
import nltk
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv
load_dotenv()
nltk.download('punkt_tab')
nltk.download('punkt')
app = FastAPI()

db_url = "postgresql+psycopg://agno:agno@db/agno"
knowledge_base = None
main_url = None

class Roadmap(BaseModel):
    topics: List[str] = Field(
        ..., 
        description="List of key topics required to achieve the goal."
    )
    subtopics: List[List[str]] = Field(
        ..., 
        description="A detailed breakdown of each topic into smaller, digestible parts."
    )
    learning_sequence: List[str] = Field(
        ..., 
        description="The logical order in which the user should learn these topics."
    )
    
def extract_keywords_keybert(text, top_n = 20):
  kw_model = KeyBERT()
  keywords = kw_model.extract_keywords(text, keyphrase_ngram_range = (1, 2), stop_words = 'english', top_n = top_n)
  return [keyword[0] for keyword in keywords]
    
def get_passages(text):
  # split text into sentences
  sentences = sent_tokenize(text)

  # combine sentences into passages
  passages = []
  current_passage = ""
  total_words = sum([len(sentence.split()) for sentence in sentences])
  for sentence in sentences:
    if len(current_passage.split()) + len(sentence.split()) < int(total_words/10):  # if word limit of passage goes beyond 1/10th the length then it appends the current passage and starts a new one
        current_passage += " " + sentence
    else:
        passages.append(current_passage.strip())
        current_passage = sentence
  if current_passage:
      passages.append(current_passage.strip())

  return passages

def generate_mcq(keywords,passage):
    prompt = f"""You are a helpful assistant. You have deeo knowledge of computer science topics and are able to generate multiple choice questions based on a list of keywords and a given passage.
    Use your own domain knowledge of the subject on the list of keywords to generate only one single multiple choice question. The question should:
    - Be informative, testing foundational and deep knowledge
    - Include clear and concise questions with four options (a, b, c, d).
    - Ensure the questions are challenging and diverse in their difficulty.
    
    Additionally, the questions generated should adhere to the following format which is a python dictionary
      "ques": "Question generated", "a": "option a", "b": "option b", "c": "option c", "d": "option d", "ans": "correct option"
      Apart from this you need not write anything before or after it.
      
      For example, you may refer to the following:
    "ques":"Why does Dijkstra's algorithm fail to work correctly with negative edge weights?", "a":"Dijkstra's algorithm is designed to find the shortest path in unweighted graphs only.",
    "b":"Negative edge weights can cause the algorithm to update distances incorrectly, leading to incorrect shortest paths.", "c":"The algorithm's priority queue cannot handle negative values."
    "d":"Negative edge weights result in an infinite loop due to cyclic paths", "ans":"b"
    Please generate the a multiple choice question (MCQ) with the help of the keywords {keywords} and you may refer to the following passage for context \n {passage}"""
    generate_mcq_agent = Agent(
            model=Groq(id='llama-3.3-70b-versatile'),
            description="You make a mcq based test.",
    )
    mcq_questions = generate_mcq_agent.run(prompt)
    return mcq_questions.content

def get_all_ques(input_text):
  passages = get_passages(input_text)
  all_ques = []
  for passage in passages:
    keywords = extract_keywords_keybert(passage)
    ques = generate_mcq(keywords, passage)
    all_ques.append(ques)
  return all_ques

def Eval(text): 
  txt = ""
  flag = False
  for i in range(len(text)):
    char = text[i]
    if char == "{":
      flag = True
      txt += char
    elif flag:
      txt += char
  return eval(txt)

def get_list_of_ques(questions):
  ques = []
  for ele in questions:
    modified_ele = Eval(ele)
    ques.append(modified_ele)
  return ques[:10]

def makeKnowledgeBase(url):
    knowledge_base = PDFUrlKnowledgeBase(
    urls=[url],
    vector_db=PgVector(table_name="recipes", db_url=db_url, search_type=SearchType.hybrid, embedder=GeminiEmbedder(),),)
    knowledge_base.load(upsert=True)

@app.get("/ask")
async def ask(query: str,topic: str):
    ask_agent = Agent(
        model=Ollama(id='llama3.2'),
        description="You give answer and explain it.",
        tools=[DuckDuckGoTools()],
        instructions=["You can give examples and analogies to explain the query to the user",f"The topic the user wants to understand is {topic}"],
    )
    response = ask_agent.run(query)
    return {"response": response}

@app.get("/roadmap")
async def roadmap(edu_level: str,query:str, url:str | None = None):
    global main_url
    if url != None and main_url != url :
        makeKnowledgeBase(url)
        main_url = url
    prompt = f"""
    You are an AI roadmap generator that creates structured learning roadmaps based on a user's education level and query.
    User Education Level: {edu_level}
    User Query: {query}

    Generate a roadmap that includes:
    1. Key Topics: The main areas of knowledge required to achieve the goal.
    2. Subtopics: A detailed breakdown of each topic into smaller, digestible parts.
    3. Learning Sequence: The logical order in which the user should learn these topics.
    4. Knowledge Base: Recommended resources, explanations, or prerequisites for better understanding.

    Ensure the roadmap is structured, logically ordered, and tailored to the user's knowledge level.
    If prerequisites are necessary, suggest them before proceeding further.
    Provide references where applicable for deeper learning.
    """
    roadmap_agent = Agent(
        model=Ollama(id='llama3.2'),
        description="You write roadmaps.",
        structured_outputs=True,
        response_model=Roadmap,
        knowledge=knowledge_base,
        tools=[DuckDuckGoTools()],
        instructions=['Do not leave any field empty','try to break the topics as much as possible',"If you want you can access the internet for more information regarding the topic with duckduckgo tool"],
    )
    agent_response = roadmap_agent.run(prompt)
    return agent_response

@app.get("/teach")
async def teach(edu_level: str,topic:str,subTopic:str, url:str | None = None):
    global main_url
    if url != None and main_url != url :
        makeKnowledgeBase(url)
        main_url = url
    prompt = f"""Imagine you are a passionate teacher who genuinely loves making learning fun and relatable. 
    Your task is to explain the subtopic {subTopic} of the topic {topic} of to a student at the {edu_level} level. 
    Use a warm, conversational tone filled with honest emotion and a sprinkle of humor. 
    Incorporate creative, out-of-the-box examples that resonate with a {edu_level} student, 
    making the material feel relevant and engaging. Feel free to be candid with both positive 
    insights and constructive, real-world feedback—as if you're chatting with a friend. Your 
    explanation should be clear, detailed, and thorough, as if you're standing in front of a 
    classroom and really care about their understanding. Go ahead and break it down like you mean it!
    """
    teaching_agent = Agent(
        model=Ollama(id='llama3.2',host="http://host.docker.internal:11434"),
        description="You teach the user.",
        knowledge=knowledge_base,
        instructions=["Always start with a nice greeting","Do not end by asking anything from the user","Explain it as much as possible","You can be repetative like human","Try to give analogies to expain it better","Don't be too frank, be a good teacher"],
        tools=[DuckDuckGoTools()],
        show_tool_calls=True,
    )
    agent_response = teaching_agent.run(prompt)
    return agent_response

@app.get("/byteChase")
async def process_file(url:str, topic:str):
    cheat_sheet_prompt = f"""
 You are a skilled assistant specializing in generating unified, complete, and structured cheat sheets from multiple sources. Your task is to take the provided content and create a comprehensive, detailed, and brief cheat sheet following the criteria and formatting guidelines below:
 Additionally, please do not use any pefixes such as greetings, disclaimers, or setup text and suffixes such as closing statements, redundant information, or reminders.
  Focus Areas:
    1) Fundamental Concepts and Overview:
      * Summarize the core ideas and principles of the topic.
      * Provide a clear, concise introduction to set the context.

    2) Key Mathematical Concepts:
      * Include any relevant mathematical ideas or theories
      * If no mathematical concepts are present, skip this section and don't even mention it.

    3) Code snippets and Explanation:
      * If the topic involves programming, include essential code examples.
      * Provide a brief explanation of what the code does and its significance.
      * You may use your own domain knowledge to generate code of the topics given.
      * Skip this section if no code is relevant.

    4) Critical Defintions:
      * Include any terms or concepts that are vital to understanding this topic
      * Ensure definitons are concise and easy to understand.

    5) Important Formulas and Equations:
      * List key formulas and equations.
      * Use proper mathematical notation for clarity.
      * Id no such thing is present, skip this section and don't even mention it.

    6) Algorithms and Key Steps:
      * Describe any relevant algorithms.
      * Focus on outlining their key steps.

  Formatting Guidelines:
    1) Heading:
      * Heading of the cheatsheet should be bold and large.

    2) Additional Text:
      * Apart from the formatted cheat sheet you must not write any other text such as "Here is the cheat sheet" or that "Note: The cheat sheet is within the 500-word limit and has a clear, organized structure with concise explanations and examples."

    3) Organized Sections:
      * Use clear headings
      * Each section should have a logical flow.

    4) Mathematical Notation:
      * Use proper Markdown syntax for formulas (e.g., $E = mc^2$ for inline equations).

    5) Conciseness and Clarity:
      * Keep explanations brief but thorough.
      * Avoid unnecessary repitition.

  Output Requirements:
    1) Write the cheat sheet in markdown format.
    2) Ensure the final output should be about 800 words.
  You have to make a cheat sheet on the topic {topic}
  """
    try:
        global main_url
        if url != None and main_url != url :
          makeKnowledgeBase(url)
          main_url = url
        cheat_sheet_agent = Agent(
            model=Ollama(id='llama3.2', host="http://host.docker.internal:11434"),
            description="You make cheet sheet",
            knowledge=knowledge_base,
            tools=[DuckDuckGoTools()],
            show_tool_calls=True,
        )
        # Generate cheat sheet
        cheat_sheet = cheat_sheet_agent.run(cheat_sheet_prompt).content
        questions = get_all_ques(cheat_sheet)
        ques = get_list_of_ques(questions) 
        
        return {"cheat_sheet": cheat_sheet, "questions": ques}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    