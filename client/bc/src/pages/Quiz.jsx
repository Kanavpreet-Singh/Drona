import React, { useState, useEffect, useMemo } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import "./Quiz.css";
import logo from "../utils/bc.png";

const Quiz = () => {
    const navigate = useNavigate();
    const location = useLocation();
    const questionsProp = location.state?.questions;

    const initialTime = 600; // 10 minutes

    const [questionsState, setQuestionsState] = useState([]);
    const [currentQuestion, setCurrentQuestion] = useState(0);
    const [timer, setTimer] = useState(() => {
        const storedTime = localStorage.getItem('quiz-timer');
        return storedTime ? parseInt(storedTime, 10) : initialTime;
    });
    const [userAnswers, setUserAnswers] = useState([]);
    const [correctAnswers, setCorrectAnswers] = useState(0);

    // Fetch questions data
    useEffect(() => {
        if (Array.isArray(questionsProp) && questionsProp.length > 0) {
            setQuestionsState(questionsProp);
        } else {
            console.error("Invalid or empty questions data", questionsProp);
        }
    }, [questionsProp]);

    // Format questions
    const formattedQuestions = useMemo(() => {
        return questionsState?.map((q) => ({
            question: q.ques || '',
            options: { a: q.a, b: q.b, c: q.c, d: q.d },
            correctAnswer: q.ans || '',
        })) || [];
    }, [questionsState]);

    // Initialize userAnswers when formattedQuestions is ready
    useEffect(() => {
        if (formattedQuestions.length > 0) {
            const initialAnswers = formattedQuestions.map(() => ({
                answer: '',
                isCorrect: false,
            }));
            setUserAnswers(initialAnswers);
        }
    }, [formattedQuestions]);

    // Timer functionality
    useEffect(() => {
        if (timer <= 0) {
            handleSubmit();
            return;
        }

        localStorage.setItem('quiz-timer', timer.toString());

        const interval = setInterval(() => setTimer((prev) => prev - 1), 1000);
        return () => clearInterval(interval);
    }, [timer]);

    const handleAnswerChange = (index, optionKey) => {
        const updatedAnswers = [...userAnswers];
        updatedAnswers[index] = {
            answer: optionKey,
            answerText: formattedQuestions[index].options[optionKey], // Store answer text
            isCorrect: optionKey === formattedQuestions[index].correctAnswer,
        };
        setUserAnswers(updatedAnswers);

        setCorrectAnswers(updatedAnswers.filter((a) => a.isCorrect).length);
    };

    const handleSubmit = () => {
        const quizData = formattedQuestions.map((q, index) => ({
            question: q.question,
            options: q.options,
            correctAnswer: q.correctAnswer,
            correctAnswerText: q.options[q.correctAnswer], // Store correct answer text
            answer: userAnswers[index]?.answer || '',
            answerText: userAnswers[index]?.answerText || "Not Answered", // Store selected answer text
        }));

        localStorage.setItem('quiz-questions', JSON.stringify(quizData));
        localStorage.removeItem('quiz-timer');

        navigate('/review');
    };
    const handleNextQuestion = () => {
        if (currentQuestion < formattedQuestions.length - 1) {
            setCurrentQuestion((prev) => prev + 1);
        }
    };

    const handlePrevQuestion = () => {
        if (currentQuestion > 0) {
            setCurrentQuestion((prev) => prev - 1);
        }
    };

    const formatTime = (time) => {
        const minutes = Math.floor(time / 60);
        const seconds = time % 60;
        return `${minutes}:${seconds < 10 ? `0${seconds}` : seconds}`;
    };

    const getButtonClass = (index) => {
        const isActive = currentQuestion === index ? 'active' : '';
        const isAnswered = userAnswers[index]?.answer ? 'answered' : '';
        return `${isActive} ${isAnswered}`;
    };

    return (
        <div className="quiz-container">
            <div className="quiztop">
                <div className="logo1">
                    <Link to="/">
                        <img src={logo} alt="Logo" />
                    </Link>
                </div>
                <div className="quiz-timer" aria-live="polite">
                    <h3 className="timers">Time Remaining: {formatTime(timer)}</h3>
                </div>
            </div>


            <div className="quiz-content">
                <div className="quiz-question">
                    {formattedQuestions.length > 0 ? (
                        <>
                            <h2>Question {currentQuestion + 1}: {formattedQuestions[currentQuestion].question}</h2>
                            <div className="quiz-options">
                                {Object.entries(formattedQuestions[currentQuestion].options).map(([key, option]) => (
                                    <label key={key} className="quiz-option">
                                        <input
                                            type="radio"
                                            name={`question-${currentQuestion}`}
                                            value={key}
                                            checked={userAnswers[currentQuestion]?.answer === key}
                                            onChange={() => handleAnswerChange(currentQuestion, key)}
                                        />
                                        {option}
                                    </label>
                                ))}
                            </div>
                        </>
                    ) : (
                        <p>Loading questions...</p>
                    )}
                </div>
                <div className="quiz-navigation">
                    <h3 className='m'>Questions</h3>
                    <div className="question-numbers">
                        {formattedQuestions.map((_, index) => (
                            <button
                                key={index}
                                onClick={() => setCurrentQuestion(index)}
                                className={getButtonClass(index)}
                            >
                                {index + 1}
                            </button>
                        ))}
                    </div>
                    <div className="quiz-nav-buttons">
                        <button
                            onClick={handlePrevQuestion}
                            disabled={currentQuestion === 0}
                        >
                            Previous
                        </button>
                        <button
                            onClick={handleNextQuestion}
                            disabled={currentQuestion === formattedQuestions.length - 1}
                        >
                            Next
                        </button>
                    </div>
                </div>
            </div>
            <div className="quiz-submit">
                <button
                    onClick={handleSubmit}
                    disabled={userAnswers.some((a) => !a.answer) || timer <= 0}
                >
                    Submit
                </button>
            </div>
        </div>
    );
};

export default Quiz;
