# Frontend - Diabetes Knowledge Management

React-based frontend for browsing and querying diabetes clinical guidelines with hierarchical navigation and RAG-powered chat interface.

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

### Key Components

- **`App.jsx`** - Root component managing routing, layout, and state
  - Routes: `/` (HomePage) and `/guidelines/*` (DocumentViewer)
  - Manages sidebar, chat interface, and responsive behavior

- **Header** - Search bar that opens chat interface with query
- **Sidebar** - Hierarchical navigation tree from document structure
- **DocumentViewer** - Renders markdown content with images and tables
- **ChatInterface** - RAG-powered chat connected to backend API

### Data Flow

1. **Document Loading**: `useDocument` hook loads `src/data/document_structure.json`
2. **Navigation**: Sidebar → React Router → DocumentViewer → MarkdownRenderer
3. **Chat**: Header search → ChatInterface → `chatService.js` → Backend API

### Backend Integration

The frontend connects to the backend via `src/services/chatService.js`:

- **Endpoint**: `POST /api/chat` (streaming responses)
- **Session Management**: Optional `session_id` for conversation continuity
- **Environment**: Set `VITE_API_BASE_URL` in `.env` (defaults to `http://localhost:8000`)

## Project Structure

```
frontend/
├── src/
│   ├── App.jsx              # Main app component
│   ├── components/
│   │   ├── chat/            # Chat interface components
│   │   ├── content/         # Document rendering
│   │   ├── layout/          # Header, Sidebar
│   │   └── navigation/      # Navigation tree
│   ├── hooks/
│   │   ├── useDocument.js   # Document data loading
│   │   └── useChat.js       # Chat state management
│   ├── services/
│   │   └── chatService.js   # Backend API integration
│   └── data/
│       └── document_structure.json  # Static document hierarchy
├── public/
│   └── images/              # 64 PNG images from source PDF
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

The frontend is a static React app that loads document structure from JSON and connects to the backend API for RAG-powered chat functionality.
