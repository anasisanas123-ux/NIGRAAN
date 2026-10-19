const Student = require("../models/Student");
const path = require("path");
const fs = require("fs");
const pythonEngine = require('../utils/pythonEngine');

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

exports.enrollStudent = async (req, res) => {
  try {
    const studentId = String(req.params.id || '').trim();
    if (!/^[A-Za-z0-9_-]{1,64}$/.test(studentId)) {
      return res.status(400).json({ success: false, error: 'Student ID may contain only letters, numbers, hyphens, or underscores.' });
    }
    const student = await Student.findOne({ student_id: studentId });
    if (!student) return res.status(404).json({ success: false, error: 'Create the student record before enrolling face photos.' });
    const images = req.body?.images;
    if (!Array.isArray(images) || images.length < 1 || images.length > 12) {
      return res.status(400).json({ success: false, error: 'Capture between 1 and 12 face photos.' });
    }
    const folderName = `${studentId}__${student.name.trim().replace(/[^A-Za-z0-9_-]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 80) || 'Person'}`;
    const imageDir = path.join(__dirname, '..', '..', '..', 'data', 'raw_images', folderName);
    await fs.promises.mkdir(imageDir, { recursive: true });
    const saved = [];
    for (let i = 0; i < images.length; i += 1) {
      const match = typeof images[i] === 'string' && images[i].match(/^data:image\/(jpeg|jpg|png|webp);base64,([A-Za-z0-9+/]+={0,2})$/i);
      if (!match || images[i].length > 1400000) {
        return res.status(400).json({ success: false, error: 'An enrollment photo is invalid or too large.' });
      }
      const mime = match[1].toLowerCase();
      const bytes = Buffer.from(match[2], 'base64');
      if (!bytes.length || bytes.length > 1024 * 1024) {
        return res.status(400).json({ success: false, error: 'Each photo must be smaller than 1 MB.' });
      }
      const extension = mime === 'png' ? 'png' : mime === 'webp' ? 'webp' : 'jpg';
      const photoPath = path.join(imageDir, `browser_${Date.now()}_${i}.${extension}`);
      await fs.promises.writeFile(photoPath, bytes);
      saved.push(photoPath);
    }
    const result = await pythonEngine.enrollIdentity({
      identity_id: studentId,
      name: student.name,
      role: 'student',
      images,
    });
    student.photo_path = saved[0] || student.photo_path;
    student.embedding_path = path.join('data', 'embeddings', `${studentId}.npy`);
    await student.save();
    res.json({ success: true, data: { ...result.data, photos_saved: saved.length }, message: 'Face enrollment is live in the camera recognition gallery.' });
  } catch (err) {
    res.status(err.statusCode === 400 ? 400 : 503).json({ success: false, error: err.message || 'Could not enroll student face photos.' });
  }
};

exports.rebuildEmbeddings = async (_req, res) => {
  try {
    const result = await pythonEngine.rebuildEmbeddings();
    const errors = result.data?.errors || [];
    res.json({
      success: true,
      data: result.data,
      message: errors.length
        ? `Refreshed ${result.data?.updated || 0} photo folder(s). ${errors.join(' | ')}`
        : `Embedding gallery refreshed (${result.data?.updated || 0} photo folder(s) processed).`,
    });
  } catch (err) {
    res.status(503).json({ success: false, error: err.message || 'Could not rebuild embeddings.' });
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
    const studentId = String(req.params.id || '').trim();
    if (!/^[A-Za-z0-9_-]{1,64}$/.test(studentId)) {
      return res.status(400).json({ success: false, error: 'Invalid student ID.' });
    }
    // Remove the identity first; otherwise a deleted student can still be
    // recognized from the persisted embedding after a camera restart.
    // Repeat cleanup even if the database record was already removed by an
    // older dashboard or a previous request stopped partway through deletion.
    try {
      await pythonEngine.removeIdentity(studentId);
    } catch (err) {
      return res.status(503).json({
        success: false,
        error: `Student was not deleted because the recognition gallery could not be updated: ${err.message}`,
      });
    }

    await Student.findOneAndDelete({ student_id: studentId });

    const rawImages = path.join(__dirname, '..', '..', '..', 'data', 'raw_images');
    try {
      const prefixes = [`${studentId}__`, `${studentId}_`];
      const folders = await fs.promises.readdir(rawImages, { withFileTypes: true });
      await Promise.all(folders
        .filter((entry) => entry.isDirectory() && prefixes.some((prefix) => entry.name.startsWith(prefix)))
        .map((entry) => fs.promises.rm(path.join(rawImages, entry.name), { recursive: true, force: true })));
    } catch (err) {
      if (err.code !== 'ENOENT') console.warn(`Could not remove saved photos for ${studentId}: ${err.message}`);
    }

    res.json({ success: true, message: "Student deleted from records and live recognition." });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
};

exports.getStudentPhoto = async (req, res) => {
  const sid = req.params.id;
  if (!sid || path.basename(sid) !== sid) {
    return res.status(400).json({ error: "Invalid student ID" });
  }

  const base = path.join(__dirname, "..", "..", "..", "data", "raw_images");
  try {
    if (!fs.existsSync(base)) {
      return res.status(404).json({ error: "No photos" });
    }

    const folders = fs
      .readdirSync(base, { withFileTypes: true })
      .filter((entry) => entry.isDirectory() && (entry.name.startsWith(`${sid}__`) || entry.name.startsWith(`${sid}_`)))
      .map((entry) => entry.name)
      .sort();

    if (folders.length === 0) {
      return res.status(404).json({ error: "No photos" });
    }

    const folder = path.join(base, folders[0]);
    const photos = fs
      .readdirSync(folder, { withFileTypes: true })
      .filter((entry) => entry.isFile() && /\.(jpg|jpeg|png|webp)$/i.test(entry.name))
      .map((entry) => entry.name)
      .sort();

    if (photos.length === 0) {
      return res.status(404).json({ error: "No photos" });
    }

    return res.sendFile(path.join(folder, photos[0]), (err) => {
      if (err && !res.headersSent) {
        res.status(err.statusCode || 500).json({ error: "Unable to send photo" });
      }
    });
  } catch (err) {
    return res.status(500).json({ error: "Unable to read student photos" });
  }
};
