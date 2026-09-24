import { NextRequest, NextResponse } from 'next/server';
import { getAllClaims, getClaimsByTag } from '@/database/queries.js';

export async function GET(request) {
  try {
    const { searchParams } = new URL(request.url);
    const tag = searchParams.get('tag');

    let claims;
    if (tag === 'all') {
      claims = await getAllClaims();
    } else if (tag === 'null' || tag === 'untagged') {
      claims = await getClaimsByTag(null);
    } else if (tag) {
      claims = await getClaimsByTag(tag);
    } else {
      claims = await getAllClaims();
    }

    return NextResponse.json(claims);
  } catch (error) {
    console.error('Error fetching claims:', error);
    return NextResponse.json({ error: 'Failed to fetch claims' }, { status: 500 });
  }
}
