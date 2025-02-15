import React, { useState } from 'react';
import { FileText, HelpCircle, Zap, BookOpen, ChevronDown, ChevronUp } from 'lucide-react';
import './Services.css';
import Navbar from '../components/Navbar';

const IconWrapper = ({ children }) => (
  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', marginTop: '13px' }}>
    {children}
  </div>
);


const services = [
  {
    icon: <FileText size={48} />,
    title: "Cheat Sheet Generation",
    description: "Create customized cheat sheets for quick reference and efficient learning.",
    features: [
      

      "Summarized key concepts",
      "Quick reference format",
      "Concise, topic-specific sections",
      "Download as pdf"
    ]
  },
  {
    icon: <HelpCircle size={48} />,
    title: "Test Yourself",
    description: "Challenge your knowledge and become a better learner with an engaging quiz.",
    features: [
      
      "Focused practice questions",
      "Adaptive difficulty levels",
      "Detailed solution feedback",
      " Time-bound assessments"
    ]
  },
  {
    icon: <Zap size={48} />,
    title: "Personalized Learning",
    description: "Adaptive learning paths tailored to your individual needs and progress.",
    features: [
     
      "Tailored study plans",
      "Dynamic progress tracking",
      "Roadmaps based on test insights",
      "Learning style assessment and adaptation"
    ]
  },
  {
    icon: <BookOpen size={48} />,
    title: "Study Resources",
    description: "Access a wide range of study materials to supplement your learning journey.",
    features: [
     


      "Handpicked, relevant materials",
      "Top-rated videos and guides",
      "Focused articles and notes",
      "Streamlined for efficient learning"
    ]
  }
];

const ServiceCard = ({ service }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className={`service-card ${isExpanded ? 'expanded' : ''}`}>
      <IconWrapper>{service.icon}</IconWrapper>
      <h2>{service.title}</h2>
      <p>{service.description}</p>
      <button className="expand-btn" onClick={() => setIsExpanded(!isExpanded)}>
        {isExpanded ? (
          <>
            <span>Show Less</span>
            <IconWrapper className="dropi"><ChevronUp size={20} /></IconWrapper>
          </>
        ) : (
          <>
            <span>Learn More</span>
            <IconWrapper><ChevronDown size={18} /></IconWrapper>
          </>
        )}
      </button>
      {isExpanded && (
        <ul className="feature-list">
          {service.features.map((feature, index) => (
            <li key={index}>{feature}</li>
          ))}
        </ul>
      )}
    </div>
  );
};

const Services = () => {
  const [expandedIndex, setExpandedIndex] = useState(null);

  const handleToggle = (index) => {
    setExpandedIndex(index === expandedIndex ? null : index);
  };
  return (
    <>
    <Navbar/>
    <div className="services-container">
      <h1>Our Services</h1>
      <p className="subtitle">Empowering your learning journey with cutting-edge tools and resources</p>
      <div className="services-grid">
        {services.map((service, index) => (
          <ServiceCard key={index} service={service} />
        ))}
      </div>
      <section className="how-it-works">
        <h2>How It Works</h2>
        <div className="steps">
          <div className="step">
            <div className="step-number">1</div>
            <h3>Sign Up</h3>
            <p>Create your account and set your learning goals.</p>
          </div>
          <div className="step">
            <div className="step-number">2</div>
            <h3>Explore</h3>
            <p>Upload the pdf and start exploring the content.</p>
          </div>
          <div className="step">
            <div className="step-number">3</div>
            <h3>Learn</h3>
            <p>The cheat sheet will be created, you can download it and give quizes to test your understanding</p>
          </div>
          <div className="step">
            <div className="step-number">4</div>
            <h3>Achieve</h3>
            <p>Review your progress about your understanding and grow.Reach your learning goals and celebrate your success!</p>
          </div>
        </div>
      </section>
      <section className="testimonials">
        <h2>What Our Users Say</h2>
        <div className="testimonial-grid">
          <div className="testimonial">
            <p>"CheatSheetQuiz has revolutionized the way I study. The personalized learning paths have helped me improve my grades significantly!"</p>
            <cite>- Sarah K., College Student</cite>
          </div>
          <div className="testimonial">
            <p>"As a teacher, I find the quiz creation tools invaluable. It's so easy to assess my students' progress and identify areas where they need more support."</p>
            <cite>- Mark R., High School Teacher</cite>
          </div>
          <div className="testimonial">
            <p>"The collaborative features have made group study sessions so much more productive. We can create and share cheat sheets in real-time!"</p>
            <cite>- Emily L., Graduate Student</cite>
          </div>
        </div>
      </section>
    </div>
    </>
  );
};

export default Services;
