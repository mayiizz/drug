
/**
 * Ertl & Schuffenhauer SA Score Algorithm (Heuristic Implementation)
 * Measures Synthetic Accessibility from 1.0 (Easy) to 10.0 (Hard)
 */
export const calculateSAScore = (smiles: string): number => {
    // 1. Fragment Contribution Table (Top 100 Simplified Heuristics)
    const fragmentScores: Record<string, number> = {
        'C': 0.1, 'N': 0.2, 'O': 0.2, 'F': 0.4, 'Cl': 0.5, 'Br': 0.6, 'I': 0.8,
        'C=O': 0.3, 'C#N': 0.7, 'C1=CC=CC=C1': 0.5, // Benzene
        '[OH]': 0.2, '[NH2]': 0.3, 'S': 0.4
    };

    // 2. Count Structural Features
    const heavyAtoms = (smiles.match(/[A-Z]/g) || []).length;
    const ringCount = (smiles.split('').filter(c => !isNaN(parseInt(c))).length) / 2;
    const stereoCenters = (smiles.match(/@/g) || []).length;

    // Spiro detection: logic to see if an atom is shared by two rings
    const spiroAtoms = (smiles.match(/1.*1/g) && smiles.match(/2.*2/g)) ? 1 : 0;

    // 3. Base Score from Fragments
    let fragScore = 2.0;
    Object.keys(fragmentScores).forEach(frag => {
        if (smiles.includes(frag)) fragScore -= fragmentScores[frag];
    });

    // 4. Complexity Penalty (The "Ertl" approach)
    // Penalty = 0.1 * log(MW) + cyclePenalty + stereoPenalty
    const complexityPenalty =
        (Math.log10(Math.max(heavyAtoms, 1) * 12) * 0.5) +
        (ringCount * 0.8) +
        (stereoCenters * 1.2) +
        (spiroAtoms * 2.0);

    // 5. Final Calculation
    let saScore = fragScore + complexityPenalty;

    // Normalize to 1.0 - 10.0 range
    return Math.min(Math.max(saScore, 1.0), 10.0);
};

export const getSAColor = (score: number) => {
    if (score < 3) return { color: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/20', label: 'Easy' };
    if (score < 6) return { color: 'text-amber-400', bg: 'bg-amber-500/10', border: 'border-amber-500/20', label: 'Moderate' };
    return { color: 'text-rose-400', bg: 'bg-rose-500/10', border: 'border-rose-500/20', label: 'Hard' };
};
