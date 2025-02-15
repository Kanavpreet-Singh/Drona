import React from 'react'
import { FileText, HelpCircle, Zap, BookOpen } from 'lucide-react';
import './AboutUs.css';
import kaavyaimg from '../utils/WhatsApp Image 2025-01-11 at 19.54.02_859e7929.jpg'
import bhuviimg from '../utils/WhatsApp Image 2025-01-12 at 23.37.36_8e68fca7.jpg'
import manrajimg from '../utils/WhatsApp Image 2025-01-14 at 16.35.11_9e4a1033.jpg'
import sidimg from '../utils/WhatsApp Image 2025-01-14 at 16.38.52_3d7039b1.jpg'
import Navbar from '../components/Navbar';
const teamMembers = [
    {
      name: 'Kaavya Sidher',
      role: 'Lead Developer',
      image: kaavyaimg,
      description: 'Kaavya, our lead developer, leverages his tech expertise to bring our vision to life.',
    },
    {
      name: 'Manraj Singh Cheema',
      role: 'Backend developer',
      image: manrajimg,
      description: 'Manraj, the technical genius, builds the foundation of our project and ensures everything runs smoothly.',
    },
    {
      name: 'Bhuvi',
      role: 'Backend Developer',
      image: bhuviimg,
      description: 'Bhuvi curates high-quality content, driving efficiency and smooth execution.',
    },
    {
      name: 'Siddarth Arora',
      role: 'Frontend Developer',
      image: sidimg,
      description: 'Siddarth designs an engaging and seamless user experience.',
    },
  ];
const AboutUs = () => {
  return (
    <>
    <Navbar />
    <div className="about-us-container">
      
      <section className="mission-section">
        <h2>Our Mission</h2>
        <p>At ByteChase, Perfectly summarized cheat sheets packed with every bit of information you need—paired with interactive quizzes—provide a seamless, easy, and highly efficient learning experience. At ByteChase, we’re redefining how you learn, offering tools that make mastering concepts faster and more convenient than ever.</p>
      </section>
      <section className="services-section">
        <h2>Our Services</h2>
        <div className="services-grid">
          <div className="service-item">
            <FileText size={36} />
            <h3>Cheat Sheet Generation</h3>
            <p>Create concise, structured cheat sheets to make extracting information effortless. Our AI-powered system provides with a simplified version of complex topics in easy-to-digest formats.</p>
          </div>
          <div className="service-item">
            <HelpCircle size={36} />
            <h3>Interactive Quizzes</h3>
            <p>Test your knowledge with our adaptive quizes designed to deepen your understanding and provide valuable insights into any topic. Our system dynamically adjusts question difficulty based on your performance, ensuring you're consistently challenged and progressing.</p>
          </div>
          <div className="service-item">
            <Zap size={36} />
            <h3>Personalized Learning Paths</h3>
            <p>Our algorithm analyzes your strengths and weaknesses to create customized learning paths, helping you focus on areas that need improvement.</p>
          </div>
          <div className="service-item">
            <BookOpen size={36} />
            <h3>Comprehensive Study Resources</h3>
            <p>Access a wide range of study materials tailored to your topic, including video tutorials, practice problems, and in-depth articles. ByteChase simplifies finding the resources you need, ensuring a comprehensive and efficient learning journey.</p>
          </div>
        </div>
      </section>
      <section className="team-section">
        <h2>Meet Our Team</h2>
        <div className="team-grid">
          {teamMembers.map((member, index) => (
            <div key={index} className="team-member">
              <img src={member.image} alt={member.name} />
              <h3>{member.name}</h3>
              <p className="role">{member.role}</p>
              <p className="description">{member.description}</p>
            </div>
          ))}
        </div>
      </section>
    </div>  
    </>
  )
}

export default AboutUs
