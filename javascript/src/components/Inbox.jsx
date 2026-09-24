'use client';

import styles from './Inbox.module.css';
import ClaimComponent from '@/components/Claim.jsx';

export default function Inbox({ inboxId, onSelectClaim, selectedClaimId, claims, loading, onTagClick }) {

    const getInboxTitle = () => {
        if (inboxId === 'all') return 'All Claims';
        if (inboxId === 'untagged') return 'Untagged Claims';
        return inboxId.charAt(0).toUpperCase() + inboxId.slice(1) + ' Claims';
    };

    if (loading) {
        return (
            <div className={styles.container}>
                <div className="header">
                    <h2>{getInboxTitle()}</h2>
                </div>
                <div className={styles.loading}>Loading claims...</div>
            </div>
        );
    }

    return (
        <div className={styles.container}>
            <div className={styles.header}>
                <h2>{getInboxTitle()}</h2>
            </div>
            <div className={styles.claimsList}>
                {claims.length === 0 ? (
                    <div className={styles.empty}>No claims in this inbox</div>
                ) : (
                    claims.map((claim) => (
                        <ClaimComponent
                            key={claim.id}
                            claim={claim}
                            isSelected={selectedClaimId === claim.id}
                            onClick={() => onSelectClaim(claim)}
                            onTagClick={onTagClick}
                        />
                    ))
                )}
            </div>
        </div>
    );
}
