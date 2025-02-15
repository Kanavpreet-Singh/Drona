const { google } = require('googleapis');
const fs = require('fs');
const path = require('path');
const multer = require('multer');
const tmp = require('tmp'); // Temporary file creation

// Google OAuth2 setup (same as before)
const CLIENT_ID = process.env.GOOGLE_CLIENT_ID;
const CLIENT_SECRET = process.env.GOOGLE_CLIENT_SECRET;
const REDIRECT_URI = "https://developers.google.com/oauthplayground";
const REFRESH_TOKEN = process.env.GOOGLE_REFRESH_TOKEN;

const oauth2Client = new google.auth.OAuth2(CLIENT_ID, CLIENT_SECRET, REDIRECT_URI);
oauth2Client.setCredentials({ refresh_token: REFRESH_TOKEN });

const drive = google.drive({ version: 'v3', auth: oauth2Client });

// Set up multer storage (in-memory storage for file processing)
const storage = multer.memoryStorage();
const upload = multer({ storage: storage }).single('pdf'); // Expecting a field named 'pdf'

// Middleware to handle file upload
function uploadFileMiddleware(req, res, next) {
    upload(req, res, function (err) {
        if (err) {
            return res.status(500).send('Error uploading file: ' + err.message);
        }
        next();
    });
}

/**
 * Create a folder in Google Drive
 * @param {string} folderName
 * @returns {string} Folder ID
 */
async function createFolder(folderName) {
    const fileMetadata = {
        name: folderName,
        mimeType: 'application/vnd.google-apps.folder',
    };
    const folder = await drive.files.create({
        resource: fileMetadata,
        fields: 'id',
    });
    return folder.data.id;
}

/**
 * Upload a file to Google Drive from memory storage
 * @param {string} folderId
 * @param {object} file
 * @returns {string} File URL
 */
async function uploadFile(folderId, file) {
    // Check if the file exists and handle it properly
    if (!file || !file.buffer) {
        throw new Error("Invalid file. Buffer not found.");
    }

    // Create a temporary file to store the file buffer
    const tmpFile = tmp.fileSync({ postfix: ".pdf" }); // Temporary file with .pdf extension

    // Write the buffer to the temporary file
    fs.writeFileSync(tmpFile.name, file.buffer);

    console.log("File exists:", tmpFile.name);

    // Upload the file to Google Drive
    const response = await drive.files.create({
        requestBody: {
            name: file.originalname,
            parents: [folderId], // Folder ID where the file should be uploaded
        },
        media: {
            mimeType: file.mimetype,
            body: fs.createReadStream(tmpFile.name), // Create a readable stream from the temporary file
        },
        fields: 'id',
    });

    // Make the file publicly accessible
    await drive.permissions.create({
        fileId: response.data.id,
        requestBody: {
            role: 'reader',
            type: 'anyone',
        },
    });

    // Generate public URL
    const fileUrl = `https://drive.google.com/uc?id=${response.data.id}`;
    
    // Clean up the temporary file after upload
    tmpFile.removeCallback();

    return fileUrl;
}

module.exports = { createFolder, uploadFile, uploadFileMiddleware };
