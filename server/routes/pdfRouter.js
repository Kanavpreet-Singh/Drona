const express = require('express');
const multer = require('multer');
const router = express.Router();
const { uploadPdfToDrive } = require('../controllers/pdfController');
const { verifyToken } = require('../middlewares/tokenValidation');
const pdfValidation = require('../middlewares/pdfValidation');

// Multer setup for memory storage
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

// Upload PDF to Google Drive
router.post('/upload', verifyToken, upload.single('pdf'), pdfValidation, uploadPdfToDrive);

module.exports = router;
