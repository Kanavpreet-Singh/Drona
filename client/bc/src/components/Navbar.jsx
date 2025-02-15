import React, { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import logo from "../utils/bc.png";
import { isAuthenticated as checkAuth, signOut } from "../utils/auth"; // Import functions from auth.js
import "./Navbar.css";

const Navbar = () => {
  const [click, setClick] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false); // Fixed typo in variable name
  const navigate = useNavigate();

  const handleClick = () => setClick(!click);
  const closeMobileMenu = () => setClick(false);

  const handleLogout = () => {
    signOut(); // Call the signOut function from auth.js
    setIsAuthenticated(false); // Update state immediately
    navigate("/login");
  };

  useEffect(() => {
    const checkUserAuthStatus = async () => {
      const isUserAuthenticated = await checkAuth(); // Check if the user is authenticated
      setIsAuthenticated(isUserAuthenticated); // Update state based on authentication status
    };

    checkUserAuthStatus(); // Run on component mount
  }, []); // Empty dependency array means this effect runs once when the component mounts

  return (
    <nav className="nav">
      <div className="logo">
        <Link to="/" onClick={closeMobileMenu}>
          <img src={logo} alt="Logo" />
        </Link>
      </div>
      <ul>
        <li>
          <Link to="/" className="nav-link" onClick={closeMobileMenu}>
            Home
          </Link>
        </li>
        <li>
          <Link to="/About" className="nav-link" onClick={closeMobileMenu}>
            About
          </Link>
        </li>
        <li>
          <Link to="/contact" className="nav-link" onClick={closeMobileMenu}>
            Contact Us
          </Link>
        </li>
        <li>
          <Link to="/Services" className="nav-link" onClick={closeMobileMenu}>
            Services
          </Link>
        </li>
      </ul>
      <div className="buttoncontainer">
        {isAuthenticated ? (
          <button onClick={handleLogout}>Logout</button>
        ) : (
          <>
            <Link to="/signup" className="nav-link" onClick={closeMobileMenu}>
              <button>Sign Up</button>
            </Link>
            <Link to="/login" className="nav-link" onClick={closeMobileMenu}>
              <button>Sign In</button>
            </Link>
          </>
        )}
      </div>
    </nav>
  );
};

export default Navbar;
