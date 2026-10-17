
import axios from 'axios';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

const api = axios.create({
    baseURL: API_BASE,
    headers: {
        'Content-Type': 'application/json',
    },
});

// ── Global error interceptor ──────────────────────────────────────────────────
// FastAPI 422 errors return { detail: string } or { detail: [{msg, loc}] }
// This interceptor converts ALL error responses into a plain Error with a
// human-readable message — so every catch block just gets error.message
api.interceptors.response.use(
    response => response,
    error => {
        let message = 'Unknown error';

        if (error.response?.data) {
            const data = error.response.data;

            if (typeof data.detail === 'string') {
                // FastAPI validation or our ValueError
                message = data.detail;
            } else if (Array.isArray(data.detail)) {
                // Pydantic field validation errors — extract human readable text
                message = data.detail
                    .map((e: { msg?: string; loc?: string[] }) =>
                        `${e.loc?.slice(1).join(' → ') ?? 'field'}: ${e.msg ?? 'invalid'}`)
                    .join('\n');
            } else if (typeof data === 'string') {
                message = data;
            } else {
                message = JSON.stringify(data);
            }
        } else if (error.message) {
            message = error.message;
        }

        return Promise.reject(new Error(message));
    }
);

// ── Types ─────────────────────────────────────────────────────────────────────

export interface MoleculeAnalysisResponse {
    molecular_weight: number;
    logp: number;
    tpsa: number;
    h_donors: number;
    h_acceptors: number;
    rotatable_bonds?: number;
    lipinski_violations?: number;
    drug_likeness: number;
    toxicity: number;
    solubility: number;
    absorption: number;
    metabolism: number;
    image: string;
    error?: string;
}

export interface ProteinResponse {
    name: string;
    gene: string;
    uniprot_id: string;
    sequence: string;
    error?: string;
}

export interface ValidationParams {
    mol_weight: { value: number; verdict: 'PASS' | 'WARN' | 'FAIL' };
    logp: { value: number; verdict: 'PASS' | 'WARN' | 'FAIL' };
    tpsa: { value: number; verdict: 'PASS' | 'WARN' | 'FAIL' };
    rot_bonds: { value: number; verdict: 'PASS' | 'WARN' | 'FAIL' };
    h_donors: { value: number; verdict: 'PASS' | 'WARN' | 'FAIL' };
    h_acceptors: { value: number; verdict: 'PASS' | 'WARN' | 'FAIL' };
    qed: { value: number; verdict: 'PASS' | 'WARN' | 'FAIL' };
    sa_score: { value: number; verdict: 'PASS' | 'WARN' | 'FAIL' };
    novelty: { value: number; verdict: 'PASS' | 'WARN' | 'FAIL' };
}

export interface Candidate {
    smiles: string;
    /** Max P(hit) across Tox21 tasks; null if unavailable */
    activity_score?: number | null;
    /** Per-assay P(positive); values may be null if that task model was not trained */
    tox21?: Record<string, number | null> | null;
    admet_score: number;
    drug_likeness: number;
    binding_score: number;   // in nM — lower = better
    // Top-level flat fields used directly in discovery/page.tsx
    logp?: number;
    tpsa?: number;
    mol_weight?: number;
    sa_score?: number;
    qed?: number;
    admet: {
        molecular_weight: number;
        logp: number;
        tpsa: number;
        h_donors: number;
        h_acceptors: number;
        rotatable_bonds: number;
        drug_likeness: number;
        sa_score: number;
    };
    validation: {
        verdict: 'PASS' | 'WARN' | 'FAIL';
        composite_score: number;
        parameters: ValidationParams;
    };
}

export interface DiscoverResponse {
    seed: string;
    target: string;
    candidates: Candidate[];
    error?: string;   // present when pipeline returns a warning alongside results
}

// ── API calls ─────────────────────────────────────────────────────────────────

export const analyzeMolecule = async (smiles: string): Promise<MoleculeAnalysisResponse> => {
    const response = await api.post('/analyze', { smiles });
    return response.data;
};

export const findSimilarDrugs = async (smiles: string) => {
    const response = await api.post('/similar-drugs', { smiles });
    return response.data;
};

export const predictSideEffects = async (smiles: string) => {
    const response = await api.post('/side-effects', { smiles });
    return response.data;
};

// lib/api.ts  —  fetchNetwork

export const fetchNetwork = async (drug?: string, disease?: string) => {
    const response = await api.post('/network',
        { drug: drug ?? '', disease: disease ?? '' },
        { responseType: 'blob' }
    );
    return response.data;
};

export interface GraphRAGResponse {
    answer: string;
    paths: string[];
    entities: string[];
    intent?: string;
    params?: Record<string, unknown>;
}

export const askGraphRAG = async (query: string): Promise<GraphRAGResponse> => {
    const response = await api.post('/graphrag', { query });
    return response.data;
};

export const fetchProtein = async (name: string): Promise<ProteinResponse> => {
    const response = await api.get(`/protein/${name}`);
    return response.data;
};

/**
 * Run the full REINVENT + DeepPurpose drug discovery pipeline.
 *
 * @param smiles           SMILES of a known drug for the target (seed)
 * @param target           Human-readable target name e.g. "EGFR"
 * @param proteinSequence  Full amino-acid sequence from fetchProtein()
 *
 * Throws Error with human-readable message if:
 *   - SMILES is invalid
 *   - Seed drug does not bind the protein (> 1000 nM)
 *   - protein_sequence is missing
 *   - Any other pipeline failure
 */
export const discoverMolecules = async (
    smiles: string,
    target: string,
    proteinSequence: string,
): Promise<DiscoverResponse> => {
    const response = await api.post('/discover', {
        smiles,
        target,
        protein_sequence: proteinSequence,
    });
    return response.data;
};

export const validateMolecule = async (smiles: string, seedSmiles?: string) => {
    const response = await api.post('/validate', { smiles, seed_smiles: seedSmiles });
    return response.data;
};

export const exportCSV = async () => {
    const response = await api.get('/export/csv', { responseType: 'blob' });
    return response.data;
};

export const exportSDF = async () => {
    const response = await api.get('/export/sdf', { responseType: 'blob' });
    return response.data;
};

export default api;
