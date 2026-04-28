# Frontend - Diabetes Knowledge Management

React-based frontend for browsing and querying diabetes clinical guidelines with hierarchical navigation and a RAG-powered chat interface.

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The app will be available at `http://localhost:5173`

## Architecture Overview

### Technology Stack
- **React 19** - UI framework
- **Vite** - Build tool and dev server
- **React Router** - Client-side routing
- **React Markdown** - Markdown content rendering

### Key Modules (Consolidated)

- **`src/App.jsx`** - Root component managing routing, layout, and view state
- **`src/ui.jsx`** - All UI components (header, sidebar, chat, document viewer)
- **`src/state.js`** - Hooks for chat + document state
- **`src/api.js`** - Backend API calls and markdown helpers
- **`src/data/document_structure.json`** - Static document hierarchy

### Data Flow

1. **Document Loading**: `useDocument` in `state.js` loads `src/data/document_structure.json`
2. **Navigation**: Sidebar -> React Router -> DocumentViewer -> MarkdownRenderer
3. **Chat**: Header search -> ChatInterface -> `api.js` -> Backend API

### Backend Integration

The frontend connects to the backend via `src/api.js`:

- **Endpoint**: `POST /api/chat` (streaming responses)
- **Session Management**: Optional `session_id` for conversation continuity
- **Environment**: Set `VITE_API_BASE_URL` in `.env` (defaults to `http://localhost:8000`)

## Project Structure

```
frontend/
├── src/
│   ├── App.jsx
│   ├── ui.jsx
│   ├── state.js
│   ├── api.js
│   ├── App.css
│   └── data/
│       └── document_structure.json
├── public/
│   └── images/
└── package.json
```

## Available Scripts

- `npm run dev` - Start development server with HMR
- `npm run build` - Build for production
- `npm run preview` - Preview production build

## How It Works

1. **Document Navigation**: Static JSON structure defines hierarchical sections
2. **Content Display**: Markdown content rendered with images, tables, and formatting
3. **Chat Interface**: Resizable pane (33-50% width) for RAG queries
4. **Search Integration**: Header search bar opens chat with initial query
