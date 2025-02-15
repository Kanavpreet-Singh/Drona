import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Person, Email, Lock } from '@mui/icons-material';
import axios from 'axios';
import "./Signup.css";
import logo from '../utils/bc.png';

const Signup = () => {
    const [signupInfo, setSignupInfo] = useState({
        name: '',
        email: '',
        password: ''
    });
    const [error, setError] = useState(null);
    const navigate = useNavigate();

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setSignupInfo((prevState) => ({
            ...prevState,
            [name]: value
        }));
    };

    const handleSignUp = async (e) => {
        e.preventDefault();
        try {
            const response = await axios.post('http://localhost:8080/auth/signup', signupInfo);
            const { token } = response.data;
            localStorage.setItem('token', token);
            navigate('/'); // Redirect to the dashboard after signup
        } catch (err) {
            // Extract a user-friendly error message
            const errorMessage = err.response?.data?.message || 'An error occurred. Please try again.';
            setError(errorMessage);
        }
    };

    return (
        <div className="signup">
            <div className="left">
                <h1>Sign Up to Bc</h1>
                <form onSubmit={handleSignUp}>
                    <div className="form-group">
                        <label htmlFor="name">Name</label>
                        <div className="input-container">
                            <Person className="input-icon" />
                            <input
                                type="text"
                                name="name"
                                placeholder="Enter your name..."
                                value={signupInfo.name}
                                onChange={handleInputChange}
                                required
                            />
                        </div>
                    </div>
                    <div className="form-group">
                        <label htmlFor="email">Email</label>
                        <div className="input-container">
                            <Email className="input-icon" />
                            <input
                                type="email"
                                name="email"
                                placeholder="Enter your email..."
                                value={signupInfo.email}
                                onChange={handleInputChange}
                                required
                            />
                        </div>
                    </div>
                    <div className="form-group">
                        <label htmlFor="password">Password</label>
                        <div className="input-container">
                            <Lock className="input-icon" />
                            <input
                                type="password"
                                name="password"
                                placeholder="Enter your password..."
                                value={signupInfo.password}
                                onChange={handleInputChange}
                                required
                            />
                        </div>
                    </div>
                    {error && <p className="error-message">{error}</p>}
                    <button type="submit">Sign Up</button>
                </form>
            </div>
            <div className="right">
                <div className="logo">
                    <Link to="/"><img src={logo} alt="Logo" /></Link>
                </div>
                <div className="hello">
                    <h1>Welcome Back</h1>
                    <p>To keep connected with us, please login with your personal info</p>
                </div>
                <Link to="/login" className="nav-link">
                    <button>Login</button>
                </Link>
            </div>
        </div>
    );
};

export default Signup;
