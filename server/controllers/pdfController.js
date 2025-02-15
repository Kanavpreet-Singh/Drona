const { createFolder, uploadFile } = require('../services/googleDriveService');
const Pdf = require('../models/pdf');

exports.uploadPdfToDrive = async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No file uploaded' });
        }

        const userId = req.user.id; // Assumes user ID is stored in the token
        const userFolderName = `User_${userId}`;
        let userFolderId = req.user.googleDriveFolderId;

        // If folder doesn't exist, create one and save it
        if (!userFolderId) {
            userFolderId = await createFolder(userFolderName);
            // Save folder ID in user database (not implemented here)
        }

        // Upload file to Google Drive
        const fileUrl = await uploadFile(userFolderId, req.file);

        // Save PDF details to MongoDB
        const pdf = new Pdf({
            fileName: req.file.originalname,
            filePath: fileUrl,
            fileSize: req.file.size,
        });

        const savedPdf = await pdf.save();
        res.status(200).json({
            message: 'File uploaded to Google Drive successfully',
            pdfId: savedPdf._id,
            fileUrl: fileUrl,
        });
    } catch (error) {
        console.error('Error uploading to Google Drive:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
};
