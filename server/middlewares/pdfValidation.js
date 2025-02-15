// pdfValidation.js
const pdfValidation = (req, res, next) => {
    const file = req.file;

    if (!file) {
        return res.status(400).json({ error: 'No file uploaded' });
    }

    // Check if the uploaded file is a PDF
    if (file.mimetype !== 'application/pdf') {
        return res.status(400).json({ error: 'Only PDF files are allowed' });
    }

    // Check file size (maximum 5MB)
    const MAX_FILE_SIZE = 5 * 1024 * 1024;
    if (file.size > MAX_FILE_SIZE) {
        return res.status(400).json({ error: 'File size exceeds the limit of 5MB' });
    }

    // All validation passed, proceed to the next middleware/controller
    next();
};

module.exports = pdfValidation;