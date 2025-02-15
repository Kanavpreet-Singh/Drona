import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm'; // For GitHub-flavored Markdown (tables, task lists, etc.)
import { Download, BookOpen } from 'lucide-react';
import Navbar from '../components/Navbar';
import './Cheat.css';

const CheatSheet = ({ response, questions }) => {
  const [cheatSheet, setCheatSheet] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    if (response) {
      setCheatSheet(response);
    }
  }, [response]);

  const handleDownload = () => {
    const blob = new Blob([cheatSheet], { type: 'text/plain' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'cheat_sheet.txt';
    link.click();
  };

  const handleTakeQuiz = () => {
    navigate('/quiz', { state: { questions } });
  };

  return (
    <div className="cheat-sheet">
      <Navbar />
      <main className="cheat-sheet__main">
        <h1 className="cheat-sheet__title">Cheat Sheet</h1>
        <div className="cheat-sheet__content">
          {cheatSheet ? (
            <ReactMarkdown 
              remarkPlugins={[remarkGfm]} 
              className="markdown-content"
            >
              {cheatSheet}
            </ReactMarkdown>
          ) : (
            <p className="cheat-sheet__loading">Loading cheat sheet...</p>
          )}
        </div>
        <div className="cheat-sheet__actions">
          <button onClick={handleDownload} className="cheat-sheet__button cheat-sheet__button--primary">
            <Download className="cheat-sheet__button-icon" /> Download Cheat Sheet
          </button>
          <button onClick={handleTakeQuiz} className="cheat-sheet__button cheat-sheet__button--primary">
            <BookOpen className="cheat-sheet__button-icon" /> Take Quiz
          </button>
        </div>
      </main>
    </div>
  );
};

export default CheatSheet;
