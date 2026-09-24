# Claims Management System

## 🏗️ Tech Stack

- **Frontend**: Next.js 15.4.5 with React 19
- **Backend**: Next.js API Routes
- **Database**: PostgreSQL with Docker
- **Language**: TypeScript
- **Styling**: CSS Modules

## 🚀 Quick Start

### Prerequisites
- Node.js
- Docker and Docker Compose

### 1. Clone and Install
```bash
npm install
```

### 2. Database Setup
```bash
# Start PostgreSQL in Docker
npm run db:start

# Seed with sample data
npm run db:reset
```

### 3. Start Development
```bash
npm run dev
```

## 📁 Project Structure

```
src/
├── app/                   # Next.js App Router
│   └── api/               # API endpoints
├── components/            # UI components
└── database/              # Database related code
    ├── client.ts          # PostgreSQL connection
    ├── queries.ts         # Database queries
    ├── seed.ts            # Database seeding
    └── seed-data.json     # Sample data
```

## 🛠️ Available Scripts

| Command            | Description                             |
|--------------------|-----------------------------------------|
| `npm run dev`      | Start development server with Turbopack |
| `npm run db:start` | Start PostgreSQL container              |
| `npm run db:stop`  | Stop PostgreSQL container               |
| `npm run db:reset` | Reset and seed database                 |