import axios from 'axios';

// Base API URL
const API_BASE_URL = 'http://localhost:8080';

// Function to check if the user is authenticated based on the token
export const isAuthenticated = async () => {
  try {
      const token = localStorage.getItem('token');
      if (!token) return false;

      const response = await axios.get(`http://localhost:8080/auth/verify-token`, {
          headers: { Authorization: `Bearer ${token}` },
      });
      return response.status === 200;
  } catch (error) {
      console.error('Error verifying token:', error.message || error);
      return false;
  }
};

// Function to sign out the user
export const signOut = () => {
  localStorage.removeItem('token');
  localStorage.removeItem('loggedInUser');
};
