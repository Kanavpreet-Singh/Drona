const { signup, login } = require('../controllers/authcontroller');
const { signupValidation, loginValidation } = require('../middlewares/authvalidation');
const { verifyToken } = require('../middlewares/tokenValidation');
require('../models/db');
const router = require('express').Router();

router.post('/signup', signupValidation, signup);
router.post('/login', loginValidation, login);

// New route for token verification
// In your server (Express) route handling
router.get('/verify-token', verifyToken, (req, res) => {
    res.status(200).json({ message: 'Token is valid' });
  });
  

module.exports = router;
