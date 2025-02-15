// import { useState } from 'react'
// import viteLogo from '/vite.svg'
import { Routes, Route } from "react-router-dom"; 
// import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import Login from "./pages/Login";
import "./App.css";
import React from "react";
import Signup from "./pages/Signup";
// import Cheat from "./pages/Cheat";
import Quiz from "./pages/Quiz";
import Review from "./pages/Review";
import AboutUs from "./pages/AboutUs";
import ContactUs from "./pages/ContactUs";
import Services from "./pages/Services";


function App() {
  

  return (
    <>
     
      <Routes>
      <Route path='/' element={<Home />} />
      
      <Route path='/login' element={<Login />} />
      <Route path='/signup' element={<Signup />} />
      {/* <Route path='/cheat' element={<Cheat />} /> */}
      <Route path='/quiz' element={<Quiz />} />
      <Route path='/review' element={<Review />} />
      <Route path='/About' element={<AboutUs />} />
      <Route path='/contact' element={<ContactUs />} />
      <Route path='/Services' element={<Services />} />
      
      </Routes>
      

    </>
  )
}

export default App
