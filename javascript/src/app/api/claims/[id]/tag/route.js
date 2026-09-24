import { NextRequest, NextResponse } from 'next/server';
import { setClaimTag } from '@/database/queries.js';

export async function PUT(request, { params }) {
  try {
    const body = await request.json();
    const resolvedParams = await params;
    const claimId = parseInt(resolvedParams.id);

    if (isNaN(claimId)) {
      return NextResponse.json({ error: 'Invalid claim ID' }, { status: 400 });
    }

    // `null` est une valeur METIER : c'est le retrait du tag, demande par le
    // bouton "Retirer le tag". L'ancienne garde `if (!tag)` la confondait avec
    // un champ manquant et rendait le retrait impossible. On distingue donc le
    // champ ABSENT (requete malformee) de la valeur nulle (retrait voulu).
    if (!('tag' in body)) {
      return NextResponse.json({ error: 'Tag is required' }, { status: 400 });
    }

    const tag = body.tag === '' ? null : body.tag;

    if (tag !== null && typeof tag !== 'string') {
      return NextResponse.json({ error: 'Tag must be a string or null' }, { status: 400 });
    }

    await setClaimTag(claimId, tag);

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error updating claim tag:', error);
    return NextResponse.json({ error: 'Failed to update claim tag' }, { status: 500 });
  }
}
