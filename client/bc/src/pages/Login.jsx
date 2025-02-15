import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Email, Lock } from '@mui/icons-material';
import axios from 'axios';
import "./Login.css";
import logo from '../utils/bc.png';

const Login = () => {
    const [loginInfo, setLoginInfo] = useState({
        email: '',
        password: ''
    });
    const [error, setError] = useState(null);
    const navigate = useNavigate();

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setLoginInfo((prevState) => ({
            ...prevState,
            [name]: value
        }));
    };

    const handleLogin = async (e) => {
        e.preventDefault();
        try {
            const response = await axios.post('http://localhost:8080/auth/login', loginInfo);
            const { token } = response.data;
            localStorage.setItem('token', token);
            navigate('/'); // Redirect to dashboard after login
        } catch (err) {
            console.log(err)
            setError("Invalid email or password");
        }
    };

    return (
        <div className="login">
            <div className="left">
                <div className="logo">
                        <Link to="/"><img src={logo} alt="Logo" /></Link>
                      </div>
                <div className="hello">
                    <h1>Hello, Friend</h1>
                    <p>Enter your personal details and start your journey with us</p>
                </div>
                <Link to="/signup" className="nav-link">
                    <button>Sign Up</button>
                </Link>
            </div>
            <div className="right">
                <h1>Sign In to Bc</h1>
                <form onSubmit={handleLogin}>
                    <div className="form-group">
                        <label htmlFor="email">Email</label>
                        <div className="input-container">
                            <Email className="input-icon" />
                            <input
                                type="email"
                                name="email"
                                placeholder="Enter your email..."
                                value={loginInfo.email}
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
                                value={loginInfo.password}
                                onChange={handleInputChange}
                                required
                            />
                        </div>
                    </div>
                    {error && <p className="error-message">{error}</p>}
                    <button type="submit">Login</button>
                </form>
            </div>
        </div>
    );
};

export default Login;
