import { db } from './client.js';
import { insertClaim } from './queries.js';
import seedData from './seed-data.json' with { type: 'json' };

async function createTables() {
  const query = `
    CREATE TABLE IF NOT EXISTS claims (
      id SERIAL PRIMARY KEY,
      content TEXT NOT NULL,
      tag VARCHAR(255)
    );
  `;

  await db.query(query);
}

async function dropTables() {
  const query = 'DROP TABLE IF EXISTS claims;';
  await db.query(query);
}

async function seed() {
  for (const claim of seedData) {
    await insertClaim(claim);
  }
}

async function reset() {
  await dropTables();
  await createTables();
  await seed();
}

await reset();
