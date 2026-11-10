// Folders shared with the Python engine (all under the project's data folder).
const path = require('path');

const DATA_DIR = path.resolve(__dirname, '..', '..', 'data');

module.exports = {
  ROOT_DIR: path.resolve(__dirname, '..', '..'),
  DATA_DIR,
  RECORDINGS_DIR: path.join(DATA_DIR, 'recordings'),
  UPLOADS_DIR: path.join(DATA_DIR, 'uploads'),
  SEARCH_INDEX_DIR: path.join(DATA_DIR, 'search_index'),
  SEARCH_QUERY_DIR: path.join(DATA_DIR, 'search_index', '_queries'),
  EXAMS_DIR: path.join(DATA_DIR, 'exams'),
  ENGINE_PORT: () => Number(process.env.PYTHON_ENGINE_PORT || 5001),
  SEARCH_PORT: () => Number(process.env.SEARCH_SERVICE_PORT || 5002),
};
