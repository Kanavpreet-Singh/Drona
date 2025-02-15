const usermodel = require("../models/user");
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");
require('dotenv').config();

const signup = async (req, res) => {
    try {
        const { name, email, password } = req.body;

        // console.log("Request Body:", req.body);

        if (!name || !email || !password) {
            // console.log("Missing fields:", { name, email, password });
            return res.status(400).json({ message: "Name, email, and password are required" });
        }

        // console.log("Checking if user exists...");
        const existingUser = await usermodel.findOne({ email });
        if (existingUser) {
            // console.log("User already exists:", email);
            return res.status(400).json({ message: "Email already exists" });
        }

        console.log("Hashing password...");
        const hashedPassword = await bcrypt.hash(password, 10);

        // console.log("Creating user...");
        const user = new usermodel({
            name,
            email,
            password: hashedPassword,
        });
        await user.save();

        // console.log("Generating token...");
        const token = jwt.sign(
            { userId: user._id, email: user.email },
            process.env.JWT_SECRET,
            { expiresIn: "1h" }
        );

        // console.log("User created successfully:", email);
        res.status(201).json({
            message: "User created successfully",
            token,
        });
    } catch (err) {
        console.error("Error occurred during signup:", err);
        res.status(500).json({ message: "Internal server error" });
    }
};



const login = async (req, res) => {
    try {
        const { email, password } = req.body;

        if (!email || !password) {
            return res.status(400).json({ message: "Email and password are required" });
        }

        const user = await usermodel.findOne({ email });
        if (!user) {
            return res.status(400).json({ message: "Invalid email or password" });
        }

        const isValidPassword = await bcrypt.compare(password, user.password);
        if (!isValidPassword) {
            return res.status(400).json({ message: "Invalid email or password" });
        }

        const token = jwt.sign(
            { userId: user._id, email: user.email },
            process.env.JWT_SECRET,
            { expiresIn: "1h" }
        );

        res.status(200).json({
            message: "Logged in successfully",
            token,
        });
    } catch (err) {
        res.status(500).json({ message: "Internal server error" });
    }
};

module.exports = { signup, login };
