import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

export async function GET(
  _req: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  const { id } = await context.params;
  const upstream = `https://alphafold.ebi.ac.uk/api/prediction/${encodeURIComponent(id)}`;

  const res = await fetch(upstream, {
    headers: {
      "User-Agent":
        "Mozilla/5.0 (compatible; CURA-StructureViewer/1.0; +https://localhost)",
      Accept: "application/json",
    },
    next: { revalidate: 86400 },
  });

  if (!res.ok) {
    return NextResponse.json(
      { error: `AlphaFold prediction not found (${res.status})` },
      { status: res.status }
    );
  }

  const data = await res.json();
  return NextResponse.json(data, {
    headers: { "Cache-Control": "public, max-age=86400" },
  });
}
