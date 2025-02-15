import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { isAuthenticated, signOut } from '../utils/auth';
import "./Home.css";
import Navbar from '../components/Navbar';
import axios from 'axios';
import Cheat from './Cheat';
import { FaRocket, FaLightbulb, FaClock, FaUpload, FaRobot, FaFileDownload } from 'react-icons/fa';

const Home = () => {
    const [file, setFile] = useState(null);
    const [loggedIn, setLoggedIn] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [uploadError, setUploadError] = useState(null);
    const navigate = useNavigate();
    const [response, setResponse] = useState("");
    const [questions, setQuestions] = useState("");
    const [loading, setLoading] = useState(false);

    const handleFileUpload = async (event) => {
        const selectedFile = event.target.files[0];
        setFile(selectedFile);

        if (!selectedFile) {
            setUploadError("No file selected.");
            return;
        }

        const MAX_FILE_SIZE = 5 * 1024 * 1024;
        if (selectedFile.size > MAX_FILE_SIZE || selectedFile.type !== "application/pdf") {
            setUploadError("Invalid file. Please upload a valid PDF under 5MB.");
            return;
        }

        try {
            setUploading(true);
            setUploadError(null);
            setLoading(true);

            const formData = new FormData();
            formData.append("pdf", selectedFile);

            const { data } = await axios.post('http://localhost:8080/pdf/upload', formData, {
                headers: { Authorization: `Bearer ${localStorage.getItem("token")}` },
            });

            alert(`File uploaded successfully! View it here: ${data.fileUrl}`);

            const backendResponse = await axios.post(
                'https://49e6-34-32-195-94.ngrok-free.app/',
                { fileurl: data.fileUrl }
            );

            console.log("Backend response:", backendResponse);

            setResponse(backendResponse.data.cheat_sheet);
            setQuestions(backendResponse.data.questions);
        } catch (error) {
            setUploadError(error.response?.data?.error || "Failed to upload file. Please try again.");
        } finally {
            setUploading(false);
            setLoading(false);
        }
    };

    const handleSignOut = () => {
        signOut();
        setLoggedIn(false);
        navigate("/login");
    };

    if (response.length !== 0) {
        return <Cheat response={response} questions={questions} />;
    }

    return (
        <div className={`home-container ${loading ? 'loading' : ''}`}>
            <div className={`content ${loading ? 'blurred' : ''}`}>
                <Navbar>
                    <div className="auth-buttons">
                        {loggedIn ? (
                            <button onClick={handleSignOut} className="auth-button">
                                Sign Out
                            </button>
                        ) : (
                            <button onClick={() => navigate("/login")} className="auth-button">
                                Sign In
                            </button>
                        )}
                    </div>
                </Navbar>
                <div className="title">
                    <h1 className="l">
                        Cheat <span>Sheet</span>
                    </h1>
                    <p className="o">
                        Your Ultimate Study Companion for College Success
                        <br />
                        Simply upload a chapter PDF, and we'll generate a detailed cheat sheet tailored to your needs.
                        Along with that, you'll receive a personalized mock test to help you excel in your upcoming college
                        quizzes.
                    </p>
                    <input
                        type="file"
                        accept="application/pdf"
                        id="pdf-upload"
                        style={{ display: 'none' }}
                        onChange={handleFileUpload}
                    />
                    <button
                        onClick={() => document.getElementById('pdf-upload').click()}
                        className="upload-button"
                        disabled={uploading}
                    >
                        {uploading ? "Uploading..." : "Upload PDF"}
                    </button>
                    {uploadError && <div className="upload-error">{uploadError}</div>}
                </div>
                <section className="features">
                    <h2 className="section-title">Why Choose Our Cheat Sheet Generator?</h2>
                    <div className="feature-grid">
                        <div className="feature-card">
                            <FaRocket className="feature-icon" />
                            <h3>Boost Your Productivity</h3>
                            <p>Save hours of manual note-taking and focus on what matters most - understanding the content.</p>
                        </div>
                        <div className="feature-card">
                            <FaLightbulb className="feature-icon" />
                            <h3>Tailored to Your Needs</h3>
                            <p>Our AI-powered system creates personalized cheat sheets that highlight key concepts and definitions.</p>
                        </div>
                        <div className="feature-card">
                            <FaClock className="feature-icon" />
                            <h3>Quick Turnaround</h3>
                            <p>Get your cheat sheet and mock test within minutes, not hours.</p>
                        </div>
                    </div>
                </section>
            </div>
            {loading && (
                <div className="loader-overlay">
                    <div className="loader-container">
                        <div className="flame-loader">
                            <div className="flame"></div>
                            <div className="flame"></div>
                            <div className="flame"></div>
                            <div className="flame"></div>
                            <div className="flame"></div>
                        </div>
                        <p className="loader-text">Processing the pdf, please wait...</p>
                    </div>
                </div>
            )}
        </div>
    );
};

export default Home;
