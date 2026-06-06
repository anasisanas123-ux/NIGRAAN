const Student = require("../models/Student");
const path = require("path");
const fs = require("fs");

exports.getAllStudents = async (req, res) => {
  try {
    const students = await Student.find().sort({ student_id: 1 });
    res.json({ success: true, count: students.length, data: students });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

exports.getStudent = async (req, res) => {
  try {
    const student = await Student.findOne({
      student_id: req.params.id,
    });
    if (!student)
      return res.status(404).json({ success: false, error: "Not found" });
    res.json({ success: true, data: student });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

exports.createStudent = async (req, res) => {
  try {
    const student = await Student.create(req.body);
    res.status(201).json({ success: true, data: student });
  } catch (err) {
    res.status(400).json({ success: false, error: err.message });
  }
};

exports.updateStudent = async (req, res) => {
  try {
    const student = await Student.findOneAndUpdate(
      { student_id: req.params.id },
      req.body,
      { new: true },
    );
    if (!student)
      return res.status(404).json({ success: false, error: "Not found" });
    res.json({ success: true, data: student });
  } catch (err) {
    res.status(400).json({ success: false, error: err.message });
  }
};

exports.deleteStudent = async (req, res) => {
  try {
    const student = await Student.findOneAndDelete({
      student_id: req.params.id,
    });
    if (!student)
      return res.status(404).json({ success: false, error: "Not found" });
    res.json({ success: true, message: "Student deleted" });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

exports.getStudentPhoto = async (req, res) => {
  const sid = req.params.id;
  const photoDir = path.join(
    __dirname,
    "..",
    "..",
    "..",
    "data",
    "raw_images",
    sid + "_*",
  );
  const glob = require("fs");
  const base = path.join(__dirname, "..", "..", "..", "data", "raw_images");
  const folders = fs.readdirSync(base).filter((f) => f.startsWith(sid));
  if (folders.length === 0) return res.status(404).json({ error: "No photos" });

  const folder = path.join(base, folders[0]);
  const photos = fs
    .readdirSync(folder)
    .filter((f) => f.match(/\.(jpg|jpeg|png)$/i));
  if (photos.length === 0) return res.status(404).json({ error: "No photos" });

  res.sendFile(path.join(folder, photos[0]));
};
