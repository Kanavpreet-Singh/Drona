require('dotenv').config();
const mongoose = require('mongoose');

mongoose.connect(process.env.MONGO_CONN)
    .then(() => console.log("MongoDB connected successfully"))
    .catch((err) => console.error("MongoDB connection error:", err));


module.exports = mongoose;
