require('dotenv').config({ path: require('path').join(__dirname, '../.env') });
const mongoose = require('mongoose');
const Student  = require('./models/Student');

const students = [
  { student_id: 'S001', name: 'Umair Iqbal',      department: 'Computer Science' },
  { student_id: 'S002', name: 'Anas Ahmed Rahim', department: 'Computer Science' },
  { student_id: 'S003', name: 'Abdul Basit',       department: 'Computer Science' },
  { student_id: 'S004', name: 'Ayan Iqbal',        department: 'Computer Science' },
  { student_id: 'S005', name: 'Rayan Iqbal',       department: 'Computer Science' },
  { student_id: 'S006', name: 'Maheen',            department: 'Computer Science' },
  { student_id: 'S007', name: 'Miral',             department: 'Computer Science' },
];

async function seed() {
  try {
    await mongoose.connect(process.env.MONGODB_URI);
    console.log('MongoDB connected');

    for (const s of students) {
      await Student.findOneAndUpdate(
        { student_id: s.student_id },
        {
          ...s,
          embedding_path: `data/embeddings/${s.student_id}.npy`,
          photo_path: `data/raw_images/${s.student_id}_${s.name.replace(/ /g, '_')}`,
        },
        { upsert: true, new: true }
      );
      console.log(`  Seeded: ${s.student_id} — ${s.name}`);
    }

    console.log('\nAll students seeded successfully.');
    const all = await Student.find().sort({ student_id: 1 });
    console.log(`Total in MongoDB: ${all.length} students`);
    all.forEach(st => console.log(`  ${st.student_id} | ${st.name}`));

  } catch (err) {
    console.error('Error:', err.message);
  } finally {
    await mongoose.disconnect();
    console.log('\nDone.');
  }
}

seed();