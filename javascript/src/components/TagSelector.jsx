'use client';

import { useRef, useState } from 'react';
import styles from './TagSelector.module.css';
import { ALLOWED_TAGS } from '@/constants/tags.js';
import { updateClaimTag, predictClaimTag } from '@/api-client.js';

export default function TagSelector({ claim, onTagUpdate }) {
    const [isUpdating, setIsUpdating] = useState(false);

    // Une prediction appartient a UNE reclamation. On stocke donc l'id avec
    // elle, au lieu de trois etats nus : le composant n'est pas remonte quand
    // `claim` change (aucun `key` cote parent), et un `pendingTag` nu survivait
    // au changement de reclamation -- le dropdown de B affichait la suggestion
    // faite pour A. Ici, un slot qui ne correspond pas a `claim.id` n'est
    // simplement jamais lu : la peremption devient impossible a exprimer.
    // Forme : { claimId, status: 'loading' | 'done' | 'error', tag?, message? }
    const [prediction, setPrediction] = useState(null);

    // Numero de la derniere requete lancee. Deux appels concurrents (predire
    // sur A, passer a B, predire sur B) peuvent revenir dans le desordre :
    // seul le plus recent a le droit d'ecrire, sinon la reponse la plus lente
    // ecrase la plus recente. Un ref et non un state : le modifier ne doit
    // declencher aucun render.
    const requestRef = useRef(0);

    // Etats derives : ils ne valent que pour la reclamation affichee.
    const current = prediction && prediction.claimId === claim?.id ? prediction : null;
    const isPredicting = current?.status === 'loading';
    const predictionError = current?.status === 'error' ? current.message : null;
    // Suggestion automatique, affichee mais PAS enregistree : tant que
    // l'utilisateur n'a pas choisi lui-meme, la base garde claim.tag.
    const pendingTag = current?.status === 'done' ? current.tag : null;

    const handleTagSelect = async (tag) => {
        if (!claim) return;

        setIsUpdating(true);
        try {
            await updateClaimTag(claim.id, tag);
            onTagUpdate(claim.id, tag);
            setPrediction(null);
        } catch (error) {
            console.error('Error updating tag:', error);
        } finally {
            setIsUpdating(false);
        }
    };

    const handlePredict = async () => {
        if (!claim) return;

        // On fige la cible AVANT l'await : l'utilisateur peut changer de
        // reclamation pendant l'appel, `claim` pointerait alors ailleurs.
        const claimId = claim.id;
        const requestId = requestRef.current + 1;
        requestRef.current = requestId;

        setPrediction({ claimId, status: 'loading' });
        try {
            const tag = await predictClaimTag(claim.content);
            if (requestRef.current !== requestId) return; // depasse par un clic plus recent
            setPrediction({ claimId, status: 'done', tag });
        } catch (error) {
            console.error('Error predicting tag:', error);
            if (requestRef.current !== requestId) return;
            setPrediction({ claimId, status: 'error', message: error.message });
        }
        // Pas de `finally` remettant un drapeau a zero : l'etat de chargement
        // est porte par le slot lui-meme, qu'un appel plus recent a deja
        // remplace le cas echeant.
    };

    if (!claim) {
        return (
            <div className={styles.container}>
                <div className={styles.placeholder}>
                    <h3>Select a claim to assign tags</h3>
                </div>
            </div>
        );
    }

    // Le select affiche UNIQUEMENT le tag enregistre. Y refleter `pendingTag`
    // rendait la suggestion invalidable en un clic : le DOM affichant deja la
    // valeur proposee, re-choisir cette option n'emet aucun evenement `change`
    // et `handleTagSelect` n'etait jamais appele. La suggestion vit donc dans
    // sa propre carte, avec un bouton -- un `click` part toujours, lui.
    const selectedTag = claim.tag ?? '';

    return (
        <div className={styles.container}>
            <div className={styles.header}>
                <h2>Claim #{claim.id}</h2>
            </div>

            <div className={styles.content}>
                <div className={styles.claimContent}>
                    <h3>Content</h3>
                    <p>{claim.content}</p>
                </div>

                <div className={styles.tagSection}>
                    <h4>Assign tag</h4>

                    <div className={styles.prediction}>
                        <button
                            type="button"
                            className={styles.predictButton}
                            onClick={handlePredict}
                            disabled={isPredicting || isUpdating}
                        >
                            {isPredicting && <span className={styles.spinner} />}
                            {isPredicting ? 'Analyse en cours...' : 'Auto-etiqueter'}
                        </button>

                        {predictionError && (
                            <p className={styles.predictionError} role="alert">
                                {predictionError}
                            </p>
                        )}

                        {pendingTag && !predictionError && (
                            <div className={styles.suggestionCard}>
                                <p className={styles.suggestionLabel}>
                                    Suggestion automatique :{' '}
                                    <strong className={styles.suggestionTag}>{pendingTag}</strong>
                                </p>

                                <div className={styles.suggestionActions}>
                                    <button
                                        type="button"
                                        className={styles.validateButton}
                                        onClick={() => handleTagSelect(pendingTag)}
                                        disabled={isUpdating}
                                    >
                                        Valider cette suggestion
                                    </button>

                                    <button
                                        type="button"
                                        className={styles.ignoreButton}
                                        onClick={() => setPrediction(null)}
                                        disabled={isUpdating}
                                    >
                                        Ignorer
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>

                    <div className={styles.tagSelector}>
                        <select
                            className={styles.select}
                            value={selectedTag}
                            onChange={(e) => handleTagSelect(e.target.value)}
                            disabled={isUpdating}
                        >
                            <option value="" disabled>
                                Select a tag
                            </option>

                            {ALLOWED_TAGS.map((tag) => (
                                <option key={tag} value={tag}>
                                    {tag}
                                </option>
                            ))}
                        </select>

                        {selectedTag === '' && (
                            <svg
                                className={styles.chevronIcon}
                                width="12"
                                height="12"
                                viewBox="0 0 292.4 292.4"
                                fill="none"
                                xmlns="http://www.w3.org/2000/svg"
                            >
                                <path
                                    fill="var(--foreground-muted)"
                                    d="M287 69.4a17.6 17.6 0 0 0-13-5.4H18.4c-5 0-9.3 1.8-12.9 5.4A17.6 17.6 0 0 0 0 82.2c0 5 1.8 9.3 5.4 12.9l128 127.9c3.6 3.6 7.8 5.4 12.8 5.4s9.2-1.8 12.8-5.4L287 95c3.5-3.5 5.4-7.8 5.4-12.8 0-5-1.9-9.2-5.5-12.8z"
                                />
                            </svg>
                        )}

                        {isUpdating && (
                            <div className={styles.updating}>
                                Updating...
                            </div>
                        )}
                    </div>

                    {claim.tag && (
                        <button
                            type="button"
                            className={styles.removeButton}
                            onClick={() => handleTagSelect(null)}
                            disabled={isUpdating}
                        >
                            Retirer le tag
                        </button>
                    )}
                </div>
            </div>
        </div>
    );
}
