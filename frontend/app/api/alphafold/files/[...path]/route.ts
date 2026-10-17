import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

/** Proxy AlphaFold PDB/CIF files with a browser UA (direct GCS links block bare proxies). */
export async function GET(
  _req: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  const { path } = await context.params;
  const filePath = path.join("/");
  const upstream = `https://alphafold.ebi.ac.uk/files/${filePath}`;

  const res = await fetch(upstream, {
    headers: {
      "User-Agent":
        "Mozilla/5.0 (compatible; CURA-StructureViewer/1.0; +https://localhost)",
      Accept: "*/*",
    },
    // Cache structures for a day on the edge/server
    next: { revalidate: 86400 },
  });

  if (!res.ok) {
    return NextResponse.json(
      { error: `AlphaFold file not found (${res.status})`, path: filePath },
      { status: res.status }
    );
  }

  const buf = await res.arrayBuffer();
  return new NextResponse(buf, {
    status: 200,
    headers: {
      "Content-Type": res.headers.get("content-type") ?? "chemical/x-pdb",
      "Cache-Control": "public, max-age=86400",
    },
  });
}
