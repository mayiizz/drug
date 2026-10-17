import { fetchProtein } from "@/lib/api";

const UNIPROT_ACCESSION_RE =
  /^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$/i;

export interface StructureJob {
  query: string;
  uniprotId: string;
  gene?: string;
  name?: string;
  pdbText: string;
  pdbUrl: string;
  /** Residue numbers annotated as active / binding sites in UniProt */
  activeSiteResidues: number[];
  submittedAt: string;
  status: "completed";
}

function toProxyUrl(absoluteUrl: string): string {
  try {
    const u = new URL(absoluteUrl);
    if (u.hostname.includes("alphafold.ebi.ac.uk") && u.pathname.startsWith("/files/")) {
      return `/api/alphafold/files/${u.pathname.replace(/^\/files\//, "")}`;
    }
  } catch {
    /* keep original */
  }
  return absoluteUrl;
}

/** Pull active-site / binding-site residue numbers from UniProt. */
export async function fetchActiveSiteResidues(uniprotId: string): Promise<number[]> {
  try {
    const res = await fetch(
      `/api/uniprot/uniprotkb/${uniprotId}.json?fields=ft_act_site,ft_binding,ft_site`
    );
    if (!res.ok) return [];
    const data = await res.json();
    const features: Array<{
      type?: string;
      location?: { start?: { value?: number }; end?: { value?: number } };
    }> = data?.features ?? [];

    const residues = new Set<number>();
    for (const f of features) {
      const type = (f.type ?? "").toLowerCase();
      if (!type.includes("active") && !type.includes("binding") && type !== "site") continue;
      const start = f.location?.start?.value;
      const end = f.location?.end?.value ?? start;
      if (start == null) continue;
      for (let r = start; r <= (end ?? start); r++) residues.add(r);
    }
    return Array.from(residues).sort((a, b) => a - b);
  } catch {
    return [];
  }
}

export function isUniProtAccession(value: string): boolean {
  return UNIPROT_ACCESSION_RE.test(value.trim());
}

/** Resolve a gene symbol or UniProt accession to a UniProt ID. */
export async function resolveUniProtId(query: string): Promise<{
  uniprotId: string;
  gene?: string;
  name?: string;
}> {
  const q = query.trim().toUpperCase();
  if (!q) throw new Error("Enter a UniProt accession or gene name.");

  if (isUniProtAccession(q)) {
    try {
      const protein = await fetchProtein(q);
      return {
        uniprotId: protein.uniprot_id || q,
        gene: protein.gene,
        name: protein.name,
      };
    } catch {
      return { uniprotId: q };
    }
  }

  const protein = await fetchProtein(q);
  if (!protein.uniprot_id) {
    throw new Error(`No UniProt ID found for "${q}".`);
  }
  return {
    uniprotId: protein.uniprot_id,
    gene: protein.gene,
    name: protein.name,
  };
}

async function fetchPdbText(url: string): Promise<string> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`AlphaFold structure not found (${res.status}).`);
  }
  const text = await res.text();
  if (!text.includes("ATOM") && !text.includes("HETATM")) {
    throw new Error("Downloaded file does not look like a valid PDB structure.");
  }
  return text;
}

/**
 * Load an AlphaFold PDB for a gene / UniProt accession.
 * Uses the AlphaFold DB prediction API, then falls back to known file URLs.
 */
export async function loadAlphaFoldStructure(query: string): Promise<StructureJob> {
  const submittedAt = new Date().toLocaleString();
  const { uniprotId, gene, name } = await resolveUniProtId(query);

  // Use Next.js rewrites (/api/alphafold/*) so the browser avoids CORS blocks
  let pdbUrl = `/api/alphafold/files/AF-${uniprotId}-F1-model_v6.pdb`;

  try {
    const metaRes = await fetch(`/api/alphafold/prediction/${uniprotId}`);
    if (metaRes.ok) {
      const meta = await metaRes.json();
      const entry = Array.isArray(meta) ? meta[0] : meta;
      if (entry?.pdbUrl) pdbUrl = toProxyUrl(entry.pdbUrl as string);
    }
  } catch {
    // Fall through to constructed URL
  }

  let pdbText: string;
  try {
    pdbText = await fetchPdbText(pdbUrl);
  } catch {
    const fallbacks = [
      `/api/alphafold/files/AF-${uniprotId}-F1-model_v4.pdb`,
      `/api/alphafold/files/AF-${uniprotId}-F1-model_v3.pdb`,
    ];
    let lastErr: unknown;
    pdbText = "";
    for (const fallback of fallbacks) {
      try {
        pdbText = await fetchPdbText(fallback);
        pdbUrl = fallback;
        lastErr = null;
        break;
      } catch (err) {
        lastErr = err;
      }
    }
    if (lastErr || !pdbText) {
      throw lastErr instanceof Error
        ? lastErr
        : new Error(`No AlphaFold structure found for ${uniprotId}.`);
    }
  }

  const activeSiteResidues = await fetchActiveSiteResidues(uniprotId);

  return {
    query: query.trim().toUpperCase(),
    uniprotId,
    gene,
    name,
    pdbText,
    pdbUrl,
    activeSiteResidues,
    submittedAt,
    status: "completed",
  };
}
