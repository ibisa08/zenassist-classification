import SQL from 'sql-template-strings';
import { db } from './client.js';

export async function getClaimsByTag(tag) {
  const query = tag === null
    ? SQL`SELECT * FROM claims WHERE tag IS NULL`
    : SQL`SELECT * FROM claims WHERE tag = ${tag}`;

  const result = await db.query(query);
  return result.rows;
}

export async function insertClaim(claim) {
  await db.query(SQL`
    INSERT INTO claims (content, tag) 
    VALUES (${claim.content}, ${claim.tag})
  `);
}

export async function getAllClaims() {
  const result = await db.query(SQL`SELECT * FROM claims ORDER BY id DESC`);
  return result.rows;
}

export async function setClaimTag(claimId, tag) {
  await db.query(SQL`
    UPDATE claims 
    SET tag = ${tag} 
    WHERE id = ${claimId}
  `);
}
