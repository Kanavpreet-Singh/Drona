# Cheat Sheet and Quiz Generator

This repository contains a group project that utilizes the open source model api inference using groq alibrary in python for generating a cheat sheet from input pdf as well as generating a quiz of 10 multiple choice questions from the input pdf and on keywords extracted from it.

## Features
1) Cheat Sheet Generation
   * Extracts text by pdf parsing and ocr for a chapter pdf.
   * Generates a cheat sheet.
2) Keyword Extraction
   * Natural language processing tools such as keybert used for extracting important keywords from      the input pdf text.
3) Quiz Generation
   * Generates 10 (MCQ) questions based on the pdf text and keywords extracted where one option is the right one among the four.

  ## Technology Stack
  * Language Model: llama-3.1-8b-instant, llama-3.3-70b-versatile
  * Languages : HTML, CSS, Javascript, Python
  * Libraries: NLTK, PyPDF, groq, paddleocr, keybert, ngrok
  * Frontend: HTML/CSS/JavaScript/React
  * Backend: Express.js, Node.js, cors, dotenv, jsonwebtoken, bycrypt, moongose, mongodb, joi, flask
