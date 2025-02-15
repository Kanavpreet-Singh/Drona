import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from "framer-motion";
import { CircularProgressbar, buildStyles } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';
import { FiHome, FiCheck, FiX, FiAward, FiTrendingUp, FiAlertCircle } from 'react-icons/fi';
import './Review.css';

const Review = () => {
  const navigate = useNavigate();
  const [questions, setQuestions] = useState([]);
  const [totalMarks, setTotalMarks] = useState(0);
  const [totalQuestions, setTotalQuestions] = useState(0);
  const [correctPercentage, setCorrectPercentage] = useState(0);
  const [activeQuestion, setActiveQuestion] = useState(0);
  const [performanceMessage, setPerformanceMessage] = useState('');

  useEffect(() => {
    const savedQuestions = localStorage.getItem('quiz-questions');
    if (savedQuestions) {
      const parsedQuestions = JSON.parse(savedQuestions);
      setQuestions(parsedQuestions);
      setTotalQuestions(parsedQuestions.length);
      const correct = parsedQuestions.filter(q => q.answer === q.correctAnswer).length;
      setTotalMarks(correct);
      const percentage = (correct / parsedQuestions.length) * 100;
      setCorrectPercentage(percentage);
      setPerformanceMessage(getPerformanceMessage(percentage));
    }
  }, []);

  const handleGoHome = () => {
    localStorage.removeItem('quiz-questions');
    navigate('/');
  };

  const nextQuestion = () => {
    setActiveQuestion((prev) => (prev + 1) % totalQuestions);
  };

  const prevQuestion = () => {
    setActiveQuestion((prev) => (prev - 1 + totalQuestions) % totalQuestions);
  };

  const getPerformanceMessage = (percentage) => {
    if (percentage >= 90) return "Excellent work! You've mastered this topic!";
    if (percentage >= 70) return "Great job! You're doing well, keep it up!";
    if (percentage >= 50) return "Good effort! There's room for improvement.";
    return "Keep practicing! You'll get better with time.";
  };

  return (
    <div className="review-container">
      <div className="review-content">
        <h1 className="review-title">Quiz Review</h1>
        <div className="review-layout">
          <div className="review-main-content">
            <div className="question-nav">
              <div className="nav-controls">
                <button onClick={prevQuestion} className="nav-button">&lt;</button>
                <span className="question-count">{activeQuestion + 1} / {totalQuestions}</span>
                <button onClick={nextQuestion} className="nav-button">&gt;</button>
              </div>
              <AnimatePresence mode='wait'>
                <motion.div
                  key={activeQuestion}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                  className="question-card"
                >
                  {questions[activeQuestion] && (
                    <>
                      <h3 className="question-title">Question {activeQuestion + 1}</h3>
                      <p className="question-text">{questions[activeQuestion].question}</p>
                      <div className="answer-container">
                        <div className="answer-box">
                          <p className="answer-label">Your Answer:</p>
                          <p className={`answer ${questions[activeQuestion].answer === questions[activeQuestion].correctAnswer ? 'correct' : 'incorrect'}`}>
                            {questions[activeQuestion].answerText || "Not Answered"}
                            {questions[activeQuestion].answer === questions[activeQuestion].correctAnswer ? 
                              <FiCheck className="icon" /> : 
                              <FiX className="icon" />
                            }
                          </p>
                        </div>
                        <div className="answer-box">
                          <p className="answer-label">Correct Answer:</p>
                          <p className="answer correct">
                            {questions[activeQuestion].correctAnswerText}
                            <FiCheck className="icon" />
                          </p>
                        </div>
                      </div>
                    </>
                  )}
                </motion.div>
              </AnimatePresence>
            </div>
            <div className="performance-breakdown">
              <h3 className="section-title">Performance Breakdown</h3>
              <div className="metrics-grid">
                <div className="metric-card">
                  <FiAward className="icon icon-yellow" />
                  <p className="metric-value">{totalMarks}</p>
                  <p className="metric-label">Correct Answers</p>
                </div>
                <div className="metric-card">
                  <FiTrendingUp className="icon icon-blue" />
                  <p className="metric-value">{Math.round(correctPercentage)}%</p>
                  <p className="metric-label">Accuracy</p>
                </div>
                <div className="metric-card">
                  <FiAlertCircle className="icon icon-red" />
                  <p className="metric-value">{totalQuestions - totalMarks}</p>
                  <p className="metric-label">Incorrect Answers</p>
                </div>
              </div>
            </div>
            <button 
              className="home-button"
              onClick={handleGoHome}
            >
              <FiHome className="icon" />
              Go to Home
            </button>
          </div>
          <div className="review-sidebar">
            <div className="score-card">
              <div className="circular-progress">
                <CircularProgressbar
                  value={correctPercentage}
                  text={`${Math.round(correctPercentage)}%`}
                  styles={buildStyles({
                    textColor: "#000",
                    pathColor: "#000",
                    trailColor: "#d6d6d6"
                  })}
                />
              </div>
              <div className="score-details">
                <p className="score-value">{totalMarks} / {totalQuestions}</p>
                <p className="score-label">Correct Answers</p>
              </div>
              <div className="performance-summary">
                <h4 className="summary-title">Performance Summary</h4>
                <p className="summary-text">{performanceMessage}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Review;

