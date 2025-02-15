
import React, { useState } from 'react';
import { Send, MapPin, Phone, Mail, Clock, Globe, CheckCircle } from 'lucide-react';
import './ContactUs.css';
import Navbar from '../components/Navbar';
const IconWrapper = (props) => <div style={{ color: '#000' }}>{props.children}</div>;
const ContactUs = () => {
    const [formStatus, setFormStatus] = useState('idle');

  const handleSubmit = (e) => {
    e.preventDefault();
    setFormStatus('sending');
    setTimeout(() => {
      setFormStatus('sent');
      setTimeout(() => setFormStatus('idle'), 3000);
    }, 1500);
  };

  return (
    <>
    <Navbar/>
    <div className="contact-us-container">
      <h1>Get in Touch</h1>
      <p className="subtitle">We'd love to hear from you. Here's how you can reach us...</p>
      <div className="contact-content">
        <div className="contact-form">
          <h2>Send us a Message</h2>
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="name">Your Name</label>
              <input type="text" id="name" placeholder="John Doe" required />
            </div>
            <div className="form-group">
              <label htmlFor="email">Your Email</label>
              <input type="email" id="email" placeholder="john@example.com" required />
            </div>
            <div className="form-group">
              <label htmlFor="subject">Subject</label>
              <input type="text" id="subject" placeholder="How can we help?" required />
            </div>
            <div className="form-group">
              <label htmlFor="message">Your Message</label>
              <textarea id="message" placeholder="Tell us more about your inquiry..." required></textarea>
            </div>
            <button type="submit" disabled={formStatus !== 'idle'}>
              {formStatus === 'idle' && (
                <>
                  Send Message <IconWrapper><Send size={18} /></IconWrapper>
                </>
              )}
              {formStatus === 'sending' && 'Sending...'}
              {formStatus === 'sent' && (
                <>
                  Sent <IconWrapper><CheckCircle size={18} /></IconWrapper>
                </>
              )}
            </button>
          </form>
        </div>
        <div className="contact-info">
          <h2 className='infok'>Contact Information</h2>
          
          <div className="info-item">
            <IconWrapper><MapPin size={24} /></IconWrapper>
            <div>
              <h3>Our Office</h3>
              <p>PEC, sector 12, chd</p>
            </div>
          </div>
          <div className="info-item">
            <IconWrapper><Phone size={24} /></IconWrapper>
            <div>
              <h3>Phone</h3>
              <p>1234567890</p>
            </div>
          </div>
          <div className="info-item">
            <IconWrapper><Mail size={24} /></IconWrapper>
            <div>
              <h3>Email</h3>
              <p>bytechase@gmail.com</p>
            </div>
          </div>
          <div className="info-item">
            <IconWrapper><Clock size={24} /></IconWrapper>
            <div>
              <h3>Working Hours</h3>
              <p>Monday - Friday: 9AM - 5PM</p>
            </div>
          </div>
          <div className="info-item">
            <IconWrapper><Globe size={24} /></IconWrapper>
            <div>
              <h3>Social Media</h3>
              <div className="social-links">
                <a href="#" aria-label="Facebook">FB</a>
                <a href="#" aria-label="Twitter">TW</a>
                <a href="#" aria-label="LinkedIn">LI</a>
                <a href="#" aria-label="Instagram">IG</a>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div className="faq-section">
        <h2>Frequently Asked Questions</h2>
        <div className="faq-grid">
          <div className="faq-item">
            <h3>How quickly can I expect a response?</h3>
            <p>We strive to respond to all inquiries within 24 hours during business days.</p>
          </div>
          <div className="faq-item">
            <h3>Do you offer phone support?</h3>
            <p>Yes, we offer phone support during our working hours for urgent matters.</p>
          </div>
          <div className="faq-item">
            <h3>Can I schedule a demo of your platform?</h3>
            <p>You can request a demo through our contact form or by emailing us directly.</p>
          </div>
          <div className="faq-item">
            <h3>How can I provide feedback on your services?</h3>
            <p>We welcome feedback! You can use the contact form or email us with your thoughts and suggestions.</p>
          </div>
        </div>
      </div>
    </div>
    </>
  )
}

export default ContactUs
